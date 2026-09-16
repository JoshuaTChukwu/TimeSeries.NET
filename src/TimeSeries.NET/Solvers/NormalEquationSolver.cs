namespace TimeSeries.Solvers;

/// <summary>
/// Gaussian elimination with partial pivoting over the normal equations, with column
/// scaling and an optional ridge.
/// </summary>
/// <remarks>
/// <para>
/// Forming <c>X'X</c> squares the condition number of <c>X</c>, and at financial scales
/// that is a real hazard. Two cheap mitigations run on every solve: the system is scaled
/// to unit diagonal, <c>D G D</c> with <c>D = diag(1 / sqrt(G_ii))</c>, which
/// equilibrates columns that differ by orders of magnitude — a price level beside a 0/1
/// flag — and the ridge, when requested, is added on that scaled system so it means the
/// same thing for every column regardless of units.
/// </para>
/// <para>
/// The dimension here is the number of coefficients, a few dozen at most, so the cubic
/// cost of elimination is irrelevant and the simpler algorithm wins. If normal equations
/// prove insufficient in practice the same interface admits a streaming QR.
/// </para>
/// </remarks>
public sealed class NormalEquationSolver : ILinearSolver
{
    /// <summary>A solver with column scaling, no ridge, and the default pivot tolerance.</summary>
    public static NormalEquationSolver Default { get; } = new();

    /// <summary>
    /// Creates a solver.
    /// </summary>
    /// <param name="ridge">
    /// Added to each diagonal entry of the scaled system, so it is relative to each
    /// column's own scale. Zero disables it. Small positive values, around 1e-8 to 1e-4,
    /// rescue a nearly collinear design at the cost of a little bias.
    /// </param>
    /// <param name="scaleColumns">
    /// Whether to scale the system to unit diagonal before elimination. Leave on; the
    /// switch exists so the effect can be measured.
    /// </param>
    /// <param name="pivotTolerance">
    /// A pivot on the scaled system below this magnitude is treated as zero and the
    /// system as singular.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="ridge"/> is negative or <paramref name="pivotTolerance"/> is not positive.
    /// </exception>
    public NormalEquationSolver(double ridge = 0d, bool scaleColumns = true, double pivotTolerance = 1e-12)
    {
        if (!(ridge >= 0d))
        {
            throw new ArgumentOutOfRangeException(nameof(ridge), ridge, "Ridge must be zero or greater.");
        }

        if (!(pivotTolerance > 0d))
        {
            throw new ArgumentOutOfRangeException(
                nameof(pivotTolerance), pivotTolerance, "Pivot tolerance must be positive.");
        }

        Ridge = ridge;
        ScaleColumns = scaleColumns;
        PivotTolerance = pivotTolerance;
    }

    /// <summary>The ridge added to the scaled diagonal, zero when none.</summary>
    public double Ridge { get; }

    /// <summary>Whether columns are scaled to unit diagonal before elimination.</summary>
    public bool ScaleColumns { get; }

    /// <summary>The magnitude below which a pivot is treated as zero.</summary>
    public double PivotTolerance { get; }

    /// <inheritdoc />
    public bool TrySolve(
        ReadOnlySpan<double> gram,
        ReadOnlySpan<double> rhs,
        Span<double> coefficients,
        out SolveDiagnostics diagnostics)
    {
        var n = rhs.Length;

        if (gram.Length != n * n)
        {
            throw new ArgumentException(
                $"A {n}-unknown system needs a {n}x{n} matrix of {n * n} values; {gram.Length} supplied.",
                nameof(gram));
        }

        if (coefficients.Length < n)
        {
            throw new ArgumentException(
                $"Coefficient span must hold at least {n} values; {coefficients.Length} available.",
                nameof(coefficients));
        }

        if (n == 0)
        {
            diagnostics = new SolveDiagnostics
            {
                Dimension = 0,
                Succeeded = true,
                Ridge = Ridge,
                ColumnsScaled = ScaleColumns,
                FailedColumn = -1,
            };
            return true;
        }

        // Working copies: the inputs are read-only, and elimination destroys its matrix.
        var a = new double[n * n];
        var b = new double[n];
        var scale = new double[n];
        gram.CopyTo(a);
        rhs.CopyTo(b);

        for (var i = 0; i < n; i++)
        {
            var diagonal = gram[(i * n) + i];

            // A zero diagonal is an all-zero column: nothing to scale by. Left at unit
            // scale, it survives only if a ridge is present to give it a pivot.
            scale[i] = ScaleColumns && diagonal > 0d ? 1d / Math.Sqrt(diagonal) : 1d;
        }

        if (ScaleColumns)
        {
            for (var i = 0; i < n; i++)
            {
                var row = i * n;

                for (var j = 0; j < n; j++)
                {
                    a[row + j] *= scale[i] * scale[j];
                }

                b[i] *= scale[i];
            }
        }

        if (Ridge > 0d)
        {
            for (var i = 0; i < n; i++)
            {
                a[(i * n) + i] += Ridge;
            }
        }

        var smallestPivot = double.PositiveInfinity;
        var largestPivot = 0d;

        // Forward elimination with partial (row) pivoting.
        for (var column = 0; column < n; column++)
        {
            var pivotRow = column;
            var pivotMagnitude = Math.Abs(a[(column * n) + column]);

            for (var row = column + 1; row < n; row++)
            {
                var candidate = Math.Abs(a[(row * n) + column]);

                if (candidate > pivotMagnitude)
                {
                    pivotMagnitude = candidate;
                    pivotRow = row;
                }
            }

            if (pivotMagnitude < PivotTolerance)
            {
                diagnostics = new SolveDiagnostics
                {
                    Dimension = n,
                    Succeeded = false,
                    SmallestPivot = 0d,
                    LargestPivot = largestPivot,
                    Ridge = Ridge,
                    ColumnsScaled = ScaleColumns,
                    FailedColumn = column,
                };
                return false;
            }

            if (pivotRow != column)
            {
                SwapRows(a, n, pivotRow, column);
                (b[pivotRow], b[column]) = (b[column], b[pivotRow]);
            }

            var pivot = a[(column * n) + column];
            smallestPivot = Math.Min(smallestPivot, pivotMagnitude);
            largestPivot = Math.Max(largestPivot, pivotMagnitude);

            for (var row = column + 1; row < n; row++)
            {
                var factor = a[(row * n) + column] / pivot;

                if (factor == 0d)
                {
                    continue;
                }

                for (var j = column; j < n; j++)
                {
                    a[(row * n) + j] -= factor * a[(column * n) + j];
                }

                b[row] -= factor * b[column];
            }
        }

        // Back substitution, un-scaling as each coefficient is finished.
        for (var i = n - 1; i >= 0; i--)
        {
            var sum = b[i];

            for (var j = i + 1; j < n; j++)
            {
                sum -= a[(i * n) + j] * coefficients[j] / scale[j];
            }

            coefficients[i] = sum / a[(i * n) + i] * scale[i];
        }

        diagnostics = new SolveDiagnostics
        {
            Dimension = n,
            Succeeded = true,
            SmallestPivot = smallestPivot,
            LargestPivot = largestPivot,
            Ridge = Ridge,
            ColumnsScaled = ScaleColumns,
            FailedColumn = -1,
        };
        return true;
    }

    private static void SwapRows(double[] a, int n, int first, int second)
    {
        var firstStart = first * n;
        var secondStart = second * n;

        for (var j = 0; j < n; j++)
        {
            (a[firstStart + j], a[secondStart + j]) = (a[secondStart + j], a[firstStart + j]);
        }
    }
}
