using TimeSeries.Solvers;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Estimation;

/// <summary>
/// The architectural claim: both Hannan-Rissanen stages solved from one Gram matrix give
/// the same coefficients as two ordinary least-squares regressions run in sequence over
/// materialised design matrices. If this test fails, the one-pass derivation is wrong.
/// </summary>
public class OnePassIdentityTests
{
    private sealed record NaiveHr(double Intercept, double[] Phi, double[] Theta, double[] Beta, double Sigma2);

    /// <summary>
    /// Textbook Hannan-Rissanen, written from the definition against explicit design
    /// matrices, sharing nothing with the library but the linear solver.
    /// </summary>
    private static NaiveHr Naive(double[] z, double[]? x, int p, int q, int m)
    {
        var n = z.Length;
        var lagDepth = Math.Max(p, q + m);
        var r = x is null ? 0 : 1;

        // Stage one over t >= L: z_t ~ 1 + z_(t-1..t-m) + x_t
        var pilotColumns = 1 + m + r;
        var pilot = Ols(n, lagDepth, pilotColumns, (t, row) =>
        {
            row[0] = 1d;
            for (var k = 1; k <= m; k++)
            {
                row[k] = z[t - k];
            }

            if (x is not null)
            {
                row[1 + m] = x[t];
            }
        }, t => z[t]);

        // Residuals wherever the pilot has enough history, with the stage-one coefficients.
        var residual = new double[n];
        for (var t = m; t < n; t++)
        {
            var value = z[t] - pilot[0];
            for (var k = 1; k <= m; k++)
            {
                value -= pilot[k] * z[t - k];
            }

            if (x is not null)
            {
                value -= pilot[1 + m] * x[t];
            }

            residual[t] = value;
        }

        // Stage two over the same t >= L: z_t ~ 1 + z_(t-1..t-p) + e-hat_(t-1..t-q) + x_t
        var columns = 1 + p + q + r;
        var coefficients = Ols(n, lagDepth, columns, (t, row) =>
        {
            row[0] = 1d;
            for (var k = 1; k <= p; k++)
            {
                row[k] = z[t - k];
            }

            for (var j = 1; j <= q; j++)
            {
                row[p + j] = residual[t - j];
            }

            if (x is not null)
            {
                row[1 + p + q] = x[t];
            }
        }, t => z[t]);

        var rss = 0d;
        var row = new double[columns];
        for (var t = lagDepth; t < n; t++)
        {
            row[0] = 1d;
            for (var k = 1; k <= p; k++)
            {
                row[k] = z[t - k];
            }

            for (var j = 1; j <= q; j++)
            {
                row[p + j] = residual[t - j];
            }

            if (x is not null)
            {
                row[1 + p + q] = x[t];
            }

            var fitted = 0d;
            for (var c = 0; c < columns; c++)
            {
                fitted += coefficients[c] * row[c];
            }

            rss += (z[t] - fitted) * (z[t] - fitted);
        }

        return new NaiveHr(
            coefficients[0],
            coefficients.Skip(1).Take(p).ToArray(),
            coefficients.Skip(1 + p).Take(q).ToArray(),
            coefficients.Skip(1 + p + q).Take(r).ToArray(),
            rss / (n - lagDepth));
    }

    private static double[] Ols(int n, int firstRow, int columns, Action<int, double[]> fill, Func<int, double> target)
    {
        var xtx = new double[columns * columns];
        var xty = new double[columns];
        var row = new double[columns];

        for (var t = firstRow; t < n; t++)
        {
            fill(t, row);
            var y = target(t);

            for (var i = 0; i < columns; i++)
            {
                xty[i] += row[i] * y;
                for (var j = 0; j < columns; j++)
                {
                    xtx[(i * columns) + j] += row[i] * row[j];
                }
            }
        }

        var beta = new double[columns];
        Assert.True(NormalEquationSolver.Default.TrySolve(xtx, xty, beta, out _));
        return beta;
    }

    [Theory]
    [InlineData(1, 1, 4)]
    [InlineData(2, 1, 6)]
    [InlineData(1, 2, 5)]
    [InlineData(2, 2, 8)]
    [InlineData(0, 1, 3)]
    public void GramSolve_EqualsTwoSequentialRegressions_Arma(int p, int q, int m)
    {
        var z = SeriesGenerator.Arma(3_000, [0.5, 0.2], [0.4, 0.1], seed: 101, constant: 0.7);

        var naive = Naive(z, null, p, q, m);
        var fit = new ArimaModel(new ArimaOptions { Order = new(p, 0, q), PilotOrder = m }).Fit(z);

        Assert.Equal(m, fit.Diagnostics.PilotOrder);
        Assert.Equal(naive.Intercept, fit.Intercept, 1e-8);
        AssertClose(naive.Phi, fit.AutoRegressive.Span, 1e-8);
        AssertClose(naive.Theta, fit.MovingAverage.Span, 1e-8);
        Assert.Equal(naive.Sigma2, fit.InnovationVariance, 1e-8);
    }

    [Fact]
    public void GramSolve_EqualsTwoSequentialRegressions_Armax()
    {
        const int N = 3_000;
        var x = SeriesGenerator.Ar1(N, 0.5, seed: 202);
        var noise = SeriesGenerator.Arma(N, [0.6], [0.3], seed: 303);
        var z = new double[N];
        for (var t = 0; t < N; t++)
        {
            z[t] = noise[t] + (2d * x[t]) + 1d;
        }

        var naive = Naive(z, x, p: 1, q: 1, m: 5);
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1), PilotOrder = 5 })
            .Fit(z, ExogenousMatrix.FromColumn(x));

        Assert.Equal(naive.Intercept, fit.Intercept, 1e-8);
        AssertClose(naive.Phi, fit.AutoRegressive.Span, 1e-8);
        AssertClose(naive.Theta, fit.MovingAverage.Span, 1e-8);
        AssertClose(naive.Beta, fit.ExogenousCoefficients.Span, 1e-8);
        Assert.Equal(naive.Sigma2, fit.InnovationVariance, 1e-8);
    }

    [Fact]
    public void GramSolve_IsInvariantToTheSeriesLevel()
    {
        // The offset shift is algebraically exact: adding 50 000 to every observation
        // changes the intercept by 50 000 (1 - sum phi) and nothing else.
        var z = SeriesGenerator.Arma(4_000, [0.6], [0.3], seed: 404, constant: 0.5);
        var lifted = z.Select(v => v + 50_000d).ToArray();
        var options = new ArimaOptions { Order = new(1, 0, 1), PilotOrder = 6 };

        var low = new ArimaModel(options).Fit(z);
        var high = new ArimaModel(options).Fit(lifted);

        Assert.Equal(low.AutoRegressive.Span[0], high.AutoRegressive.Span[0], 1e-7);
        Assert.Equal(low.MovingAverage.Span[0], high.MovingAverage.Span[0], 1e-7);
        Assert.Equal(low.Intercept + (50_000d * (1d - low.AutoRegressive.Span[0])), high.Intercept, 1e-4);
    }

    private static void AssertClose(double[] expected, ReadOnlySpan<double> actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], tolerance);
        }
    }
}
