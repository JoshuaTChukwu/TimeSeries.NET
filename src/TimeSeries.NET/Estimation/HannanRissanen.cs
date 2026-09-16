using TimeSeries.Accumulators;
using TimeSeries.Solvers;

namespace TimeSeries;

/// <summary>
/// Both Hannan-Rissanen stages, solved from one frozen Gram matrix with no data access.
/// </summary>
/// <remarks>
/// <para>
/// Let <c>G-hat</c> be the Gram of the row augmented by a constant, so every regressor
/// either stage uses is a linear functional <c>a' v-tilde_t</c>. The pilot AR(m) is a
/// sub-matrix solve; its residual <c>e-hat_t = w' v-tilde_t</c> is a vector <c>w</c>
/// built from the pilot coefficients; and the second-stage normal equations are
/// <c>A' G-hat A</c> and <c>A' G-hat e_0</c> for the matrix <c>A</c> whose columns are the
/// second-stage regressors as functionals. One traversal, regardless of <c>p</c>,
/// <c>q</c> or <c>m</c>.
/// </para>
/// <para>
/// The optional third-stage refinement is excluded: it filters the series by the
/// estimated ARMA polynomial, which is an infinite-order operation and would cost a
/// genuine second pass.
/// </para>
/// </remarks>
internal static class HannanRissanen
{
    /// <summary>Coefficients on the shifted scale, plus what produced them.</summary>
    internal sealed class Result
    {
        internal Result(
            double intercept, double[] phi, double[] theta, double[] beta,
            int pilotOrder, double pilotIntercept, double[] pilotPhi, double[] pilotBeta,
            double residualSumOfSquares, double[] standardErrors, SolveDiagnostics solve)
        {
            Intercept = intercept;
            Phi = phi;
            Theta = theta;
            Beta = beta;
            PilotOrder = pilotOrder;
            PilotIntercept = pilotIntercept;
            PilotPhi = pilotPhi;
            PilotBeta = pilotBeta;
            ResidualSumOfSquares = residualSumOfSquares;
            StandardErrors = standardErrors;
            Solve = solve;
        }

        internal double Intercept { get; }

        internal double[] Phi { get; }

        internal double[] Theta { get; }

        internal double[] Beta { get; }

        internal int PilotOrder { get; }

        internal double PilotIntercept { get; }

        internal double[] PilotPhi { get; }

        internal double[] PilotBeta { get; }

        internal double ResidualSumOfSquares { get; }

        /// <summary>Ordered as the second-stage design: intercept (if any), AR, MA, exogenous.</summary>
        internal double[] StandardErrors { get; }

        internal SolveDiagnostics Solve { get; }

        internal int ParameterCount => StandardErrors.Length;
    }

    internal static Result Solve(GramMatrix gram, GramLayout layout, ArimaOptions options, ILinearSolver solver)
    {
        var side = layout.Augmented;
        var g = Augment(gram, layout);
        var n = gram.Count;
        var p = options.Order.P;
        var q = options.Order.Q;
        var r = layout.RegressorCount;
        var intercept = options.IncludeIntercept;
        var position = 0;

        // ---- Stage one: the pilot autoregression, order chosen by AICc from nested sub-matrices.
        var pilotOrder = 0;
        var pilotIntercept = 0d;
        var pilotPhi = Array.Empty<double>();
        var pilotBeta = Array.Empty<double>();
        double[]? residualFunctional = null;

        if (q > 0)
        {
            var (minimum, maximum) = options.PilotRange;
            var bestAicc = double.PositiveInfinity;
            double[]? bestCoefficients = null;

            for (var m = minimum; m <= maximum; m++)
            {
                var columns = PilotColumns(layout, m, intercept);
                var system = Extract(g, side, columns);
                var rhs = ExtractRhs(g, side, columns);
                var x = new double[columns.Length];

                if (!solver.TrySolve(system, rhs, x, out _))
                {
                    continue;
                }

                var rss = Math.Max(g[0] - Dot(x, rhs), 0d);
                var k = columns.Length + 1;

                if (n - k - 1 <= 0d)
                {
                    continue;
                }

                var aicc = (n * Math.Log(Math.Max(rss, double.Epsilon) / n)) + (2d * k)
                    + (2d * k * (k + 1) / (n - k - 1));

                if (aicc < bestAicc)
                {
                    bestAicc = aicc;
                    bestCoefficients = x;
                    pilotOrder = m;
                }
            }

            if (bestCoefficients is null)
            {
                throw new SingularDesignException(
                    "pilot autoregression", -1,
                    $"No pilot autoregression of order {minimum}..{maximum} could be solved: the lagged " +
                    "series and regressors are collinear. A small Ridge in ArimaOptions will regularise the fit.");
            }

            position = 0;
            residualFunctional = new double[side];
            residualFunctional[0] = 1d;

            if (intercept)
            {
                pilotIntercept = bestCoefficients[position++];
                residualFunctional[layout.Constant] = -pilotIntercept;
            }

            pilotPhi = new double[pilotOrder];
            for (var k = 1; k <= pilotOrder; k++)
            {
                pilotPhi[k - 1] = bestCoefficients[position++];
                residualFunctional[layout.Lag(k)] = -pilotPhi[k - 1];
            }

            pilotBeta = new double[r];
            for (var i = 0; i < r; i++)
            {
                pilotBeta[i] = bestCoefficients[position++];
                residualFunctional[layout.Exogenous(0, i)] = -pilotBeta[i];
            }
        }

        // ---- Stage two: every regressor as a functional of the augmented row.
        var count = (intercept ? 1 : 0) + p + q + r;
        var design = new double[count][];
        var column = 0;

        if (intercept)
        {
            design[column++] = Unit(side, layout.Constant);
        }

        for (var k = 1; k <= p; k++)
        {
            design[column++] = Unit(side, layout.Lag(k));
        }

        for (var j = 1; j <= q; j++)
        {
            design[column++] = ShiftResidual(residualFunctional!, j, layout);
        }

        for (var i = 0; i < r; i++)
        {
            design[column++] = Unit(side, layout.Exogenous(0, i));
        }

        // (G A)[a][c], then M = A' (G A) and rhs = A' G e_0.
        var ga = new double[side * count];

        for (var a = 0; a < side; a++)
        {
            var row = a * side;

            for (var c = 0; c < count; c++)
            {
                var functional = design[c];
                var sum = 0d;

                for (var b = 0; b < side; b++)
                {
                    var weight = functional[b];

                    if (weight != 0d)
                    {
                        sum += g[row + b] * weight;
                    }
                }

                ga[(a * count) + c] = sum;
            }
        }

        var normal = new double[count * count];
        var rhs2 = new double[count];

        for (var c1 = 0; c1 < count; c1++)
        {
            var functional = design[c1];
            var rhsSum = 0d;

            for (var a = 0; a < side; a++)
            {
                var weight = functional[a];

                if (weight == 0d)
                {
                    continue;
                }

                rhsSum += weight * g[a * side];

                for (var c2 = 0; c2 < count; c2++)
                {
                    normal[(c1 * count) + c2] += weight * ga[(a * count) + c2];
                }
            }

            rhs2[c1] = rhsSum;
        }

        var coefficients = new double[count];

        if (!solver.TrySolve(normal, rhs2, coefficients, out var diagnostics))
        {
            throw new SingularDesignException(
                "second stage", diagnostics.FailedColumn,
                $"The second-stage normal equations are singular at design column {diagnostics.FailedColumn} " +
                $"(columns are ordered intercept, AR 1..{p}, MA 1..{q}, exogenous 1..{r}). Two regressors are " +
                "collinear or a lag cannot be identified from this series. A small Ridge in ArimaOptions " +
                "will regularise the fit; a constant differenced series is handled separately.");
        }

        var residualSumOfSquares = Math.Max(g[0] - Dot(coefficients, rhs2), 0d);
        var standardErrors = StandardErrors(normal, count, residualSumOfSquares / n, solver);

        position = 0;
        var c0 = intercept ? coefficients[position++] : 0d;
        var phi = Slice(coefficients, ref position, p);
        var theta = Slice(coefficients, ref position, q);
        var beta = Slice(coefficients, ref position, r);

        return new Result(
            c0, phi, theta, beta, pilotOrder, pilotIntercept, pilotPhi, pilotBeta,
            residualSumOfSquares, standardErrors, diagnostics);
    }

    /// <summary>
    /// <c>[[G, s], [s', N]]</c>: the Gram of the row with a trailing constant 1.
    /// </summary>
    private static double[] Augment(GramMatrix gram, GramLayout layout)
    {
        var dimension = layout.Dimension;
        var side = layout.Augmented;
        var g = new double[side * side];
        var values = gram.Values.Span;
        var sums = gram.Sums.Span;

        for (var i = 0; i < dimension; i++)
        {
            for (var j = 0; j < dimension; j++)
            {
                g[(i * side) + j] = values[(i * dimension) + j];
            }

            g[(i * side) + dimension] = sums[i];
            g[(dimension * side) + i] = sums[i];
        }

        g[(dimension * side) + dimension] = gram.Count;
        return g;
    }

    private static int[] PilotColumns(GramLayout layout, int order, bool intercept)
    {
        var columns = new int[(intercept ? 1 : 0) + order + layout.RegressorCount];
        var position = 0;

        if (intercept)
        {
            columns[position++] = layout.Constant;
        }

        for (var k = 1; k <= order; k++)
        {
            columns[position++] = layout.Lag(k);
        }

        for (var i = 0; i < layout.RegressorCount; i++)
        {
            columns[position++] = layout.Exogenous(0, i);
        }

        return columns;
    }

    /// <summary>
    /// The functional for <c>e-hat_(t-j)</c>: every series lag moves <c>j</c> deeper,
    /// every regressor moves to its lag-<c>j</c> block, the constant stays put.
    /// </summary>
    private static double[] ShiftResidual(double[] functional, int shift, GramLayout layout)
    {
        var shifted = new double[functional.Length];

        for (var k = 0; k <= layout.LagDepth - shift; k++)
        {
            shifted[layout.Lag(k + shift)] = functional[layout.Lag(k)];
        }

        for (var i = 0; i < layout.RegressorCount; i++)
        {
            shifted[layout.Exogenous(shift, i)] = functional[layout.Exogenous(0, i)];
        }

        shifted[layout.Constant] = functional[layout.Constant];
        return shifted;
    }

    private static double[] StandardErrors(double[] normal, int count, double sigma2, ILinearSolver solver)
    {
        var errors = new double[count];
        var unit = new double[count];
        var columnOfInverse = new double[count];

        for (var c = 0; c < count; c++)
        {
            Array.Clear(unit, 0, count);
            unit[c] = 1d;

            errors[c] = solver.TrySolve(normal, unit, columnOfInverse, out _)
                ? Math.Sqrt(Math.Max(sigma2 * columnOfInverse[c], 0d))
                : double.NaN;
        }

        return errors;
    }

    private static double[] Extract(double[] g, int side, int[] columns)
    {
        var k = columns.Length;
        var sub = new double[k * k];

        for (var i = 0; i < k; i++)
        {
            for (var j = 0; j < k; j++)
            {
                sub[(i * k) + j] = g[(columns[i] * side) + columns[j]];
            }
        }

        return sub;
    }

    private static double[] ExtractRhs(double[] g, int side, int[] columns)
    {
        var rhs = new double[columns.Length];

        for (var i = 0; i < columns.Length; i++)
        {
            rhs[i] = g[columns[i] * side];
        }

        return rhs;
    }

    private static double[] Unit(int length, int index)
    {
        var unit = new double[length];
        unit[index] = 1d;
        return unit;
    }

    private static double[] Slice(double[] source, ref int position, int length)
    {
        var slice = new double[length];
        Array.Copy(source, position, slice, 0, length);
        position += length;
        return slice;
    }

    private static double Dot(double[] a, double[] b)
    {
        var sum = 0d;

        for (var i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }
}
