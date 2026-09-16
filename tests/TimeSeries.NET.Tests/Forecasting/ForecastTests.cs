using TimeSeries.Solvers;
using TimeSeries.Tests.TestUtils;
using TimeSeries.Transforms;

namespace TimeSeries.Tests.Forecasting;

public class ForecastTests
{
    [Fact]
    public void RandomWalk_IntervalsAreSigmaRootH()
    {
        // The M5 acceptance criterion: ARIMA(0,1,0) standard errors equal sigma sqrt(h).
        var series = SeriesGenerator.RandomWalk(2_000, seed: 1);
        var fit = new ArimaModel(new ArimaOptions { Order = new(0, 1, 0), IncludeIntercept = false }).Fit(series);

        var forecast = fit.Forecast(ForecastHorizon.Periods(20));
        var sigma = Math.Sqrt(fit.InnovationVariance);

        for (var h = 1; h <= 20; h++)
        {
            Assert.Equal(sigma * Math.Sqrt(h), forecast.StandardError.Span[h - 1], 1e-12 * sigma);
            Assert.Equal(series[^1], forecast.Mean.Span[h - 1], 1e-9);
        }

        Assert.Equal(20, forecast.Steps);
        Assert.Equal(series.Length, forecast.Origin);
    }

    [Fact]
    public void RandomWalkWithDrift_MeanIsLastPlusHTimesDrift()
    {
        var series = SeriesGenerator.RandomWalk(2_000, seed: 2, drift: 0.3);
        var fit = new ArimaModel(new ArimaOptions { Order = new(0, 1, 0) }).Fit(series);

        var forecast = fit.Forecast(ForecastHorizon.Periods(10));

        for (var h = 1; h <= 10; h++)
        {
            Assert.Equal(series[^1] + (h * fit.Intercept), forecast.Mean.Span[h - 1], 1e-9);
        }
    }

    [Fact]
    public void Ar1_ClosedFormMeanAndIntervals()
    {
        var series = SeriesGenerator.Ar1(3_000, 0.7, seed: 3);
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).Fit(series);

        var c = fit.Intercept;
        var phi = fit.AutoRegressive.Span[0];
        var sigma2 = fit.InnovationVariance;
        var last = series[^1];
        var forecast = fit.Forecast(ForecastHorizon.Periods(12));

        for (var h = 1; h <= 12; h++)
        {
            var geometric = 0d;
            var variance = 0d;
            for (var i = 0; i < h; i++)
            {
                geometric += Math.Pow(phi, i);
                variance += Math.Pow(phi, 2 * i);
            }

            Assert.Equal((c * geometric) + (Math.Pow(phi, h) * last), forecast.Mean.Span[h - 1], 1e-10);
            Assert.Equal(Math.Sqrt(sigma2 * variance), forecast.StandardError.Span[h - 1], 1e-10);
        }
    }

    [Fact]
    public void Ma1_ForecastUsesTheLastResidualOnceThenTheMean()
    {
        var series = SeriesGenerator.Arma(3_000, [], [0.5], seed: 4, constant: 0.2);
        var fit = new ArimaModel(new ArimaOptions { Order = new(0, 0, 1) }).Fit(series);

        var c = fit.Intercept;
        var theta = fit.MovingAverage.Span[0];
        var sigma = Math.Sqrt(fit.InnovationVariance);
        var lastResidual = fit.Seed.RecentResiduals[0];
        var forecast = fit.Forecast(ForecastHorizon.Periods(5));

        Assert.Equal(c + (theta * lastResidual), forecast.Mean.Span[0], 1e-12);
        Assert.Equal(sigma, forecast.StandardError.Span[0], 1e-12);

        for (var h = 2; h <= 5; h++)
        {
            Assert.Equal(c, forecast.Mean.Span[h - 1], 1e-12);
            Assert.Equal(sigma * Math.Sqrt(1d + (theta * theta)), forecast.StandardError.Span[h - 1], 1e-12);
        }
    }

    [Fact]
    public void Arima110_IntegratesTheDifferencedRecursion()
    {
        var differenced = SeriesGenerator.Arma(2_000, [0.5], [], seed: 5, constant: 0.1);
        var series = SeriesGenerator.Integrate(differenced, 1, level: 50d);
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 1, 0) }).Fit(series);

        var forecast = fit.Forecast(ForecastHorizon.Periods(8));

        // Hand recursion on the differenced scale, then a running sum from the last level.
        var z = series[^1] - series[^2];
        var level = series[^1];
        for (var h = 1; h <= 8; h++)
        {
            z = fit.Intercept + (fit.AutoRegressive.Span[0] * z);
            level += z;
            Assert.Equal(level, forecast.Mean.Span[h - 1], 1e-9);
        }
    }

    [Fact]
    public void DeterministicSeasonalPattern_IsContinuedExactly()
    {
        // y_t = 2t + pattern[t mod 4]: (1-B)(1-B^4) annihilates it, the fit is the
        // constant zero, and the forecast must continue the pattern with zero width.
        double[] pattern = [5, -3, 1, 7];
        var series = Enumerable.Range(0, 200).Select(t => (2d * t) + pattern[t % 4]).ToArray();
        var fit = new ArimaModel(new ArimaOptions { Order = new(0, 1, 0), Seasonal = new(1, 4) }).Fit(series);

        Assert.True(fit.Diagnostics.IsConstantSeries);

        var forecast = fit.Forecast(ForecastHorizon.Periods(12));

        for (var h = 1; h <= 12; h++)
        {
            var t = 200 + h - 1;
            Assert.Equal((2d * t) + pattern[t % 4], forecast.Mean.Span[h - 1], 1e-9);
            Assert.Equal(0d, forecast.StandardError.Span[h - 1]);
        }
    }

    [Fact]
    public void Armax_HandRecursion_AndFutureRegressorsAreDifferencedFromTheTail()
    {
        const int N = 2_000;
        var x = SeriesGenerator.Ar1(N + 3, 0.5, seed: 6);
        var noise = SeriesGenerator.Arma(N, [0.5], [], seed: 7);
        var y = new double[N];
        for (var t = 0; t < N; t++)
        {
            y[t] = 1d + (t > 0 ? 0.5 * y[t - 1] : 0d) + (2d * x[t]) + (noise[t] - (t > 0 ? 0.5 * noise[t - 1] : 0d));
        }

        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) })
            .Fit(y, ExogenousMatrix.FromColumn(x.Take(N).ToArray()));
        var future = ExogenousMatrix.FromColumn(x.Skip(N).Take(3).ToArray());

        var forecast = fit.Forecast(ForecastHorizon.Periods(3), future);

        var c = fit.Intercept;
        var phi = fit.AutoRegressive.Span[0];
        var beta = fit.ExogenousCoefficients.Span[0];
        var expected1 = c + (phi * y[^1]) + (beta * x[N]);
        var expected2 = c + (phi * expected1) + (beta * x[N + 1]);
        var expected3 = c + (phi * expected2) + (beta * x[N + 2]);

        Assert.Equal(expected1, forecast.Mean.Span[0], 1e-10);
        Assert.Equal(expected2, forecast.Mean.Span[1], 1e-10);
        Assert.Equal(expected3, forecast.Mean.Span[2], 1e-10);
    }

    [Fact]
    public void Arimax_WithDifferencing_DifferencesFutureRegressorsAgainstTheLastObserved()
    {
        // y_t = beta x_t + random walk. With d = 1 and no intercept the one-step forecast
        // is y_n + beta (x_(n+1) - x_n): the future x is differenced against the retained x_n.
        const int N = 3_000;
        var x = SeriesGenerator.RandomWalk(N + 1, seed: 8);
        var walk = SeriesGenerator.RandomWalk(N, seed: 9);
        var y = new double[N];
        for (var t = 0; t < N; t++)
        {
            y[t] = (3d * x[t]) + walk[t];
        }

        var fit = new ArimaModel(new ArimaOptions { Order = new(0, 1, 0), IncludeIntercept = false })
            .Fit(y, ExogenousMatrix.FromColumn(x.Take(N).ToArray()));
        var forecast = fit.Forecast(ForecastHorizon.Periods(1), ExogenousMatrix.FromColumn([x[N]]));

        var beta = fit.ExogenousCoefficients.Span[0];
        Assert.Equal(3d, beta, 0.05);
        Assert.Equal(y[^1] + (beta * (x[N] - x[N - 1])), forecast.Mean.Span[0], 1e-9);
    }

    [Fact]
    public void Armax_FutureRegressorValidation()
    {
        var x = SeriesGenerator.Ar1(1_000, 0.5, seed: 10);
        var y = SeriesGenerator.Ar1(1_000, 0.3, seed: 11);
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).Fit(y, ExogenousMatrix.FromColumn(x));
        var horizon = ForecastHorizon.Periods(4);

        var missing = Assert.Throws<ArgumentException>(() => fit.Forecast(horizon));
        Assert.Contains("1 exogenous regressor", missing.Message, StringComparison.Ordinal);

        var wrongRows = Assert.Throws<ArgumentException>(() => fit.Forecast(horizon, ExogenousMatrix.FromColumn([1, 2, 3])));
        Assert.Contains("needs 4 rows", wrongRows.Message, StringComparison.Ordinal);
        Assert.Contains("3 supplied", wrongRows.Message, StringComparison.Ordinal);

        Assert.Throws<ArgumentException>(() => fit.Forecast(horizon, ExogenousMatrix.FromColumns([1, 2, 3, 4], [1, 2, 3, 4])));
        Assert.Throws<ArgumentNullException>(() => fit.Forecast(horizon, (ExogenousMatrix)null!));
        Assert.Throws<InvalidSeriesException>(() => fit.Forecast(horizon, ExogenousMatrix.FromColumn([1, double.NaN, 3, 4])));

        var arima = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).Fit(y);
        Assert.Throws<ArgumentException>(() => arima.Forecast(horizon, ExogenousMatrix.FromColumn([1, 2, 3, 4])));
    }

    [Fact]
    public void Intervals_AreSymmetricAndNarrowerAtLowerConfidence()
    {
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1) }).Fit(SeriesGenerator.Arma(2_000, [0.5], [0.3], seed: 12));

        var wide = fit.Forecast(ForecastHorizon.Periods(6));
        var narrow = fit.Forecast(ForecastHorizon.Periods(6), new ForecastOptions { Confidence = 0.8 });

        Assert.Equal(1.959963984540054, wide.CriticalValue, 1e-12);
        for (var h = 0; h < 6; h++)
        {
            Assert.Equal(wide.Mean.Span[h] - wide.Lower.Span[h], wide.Upper.Span[h] - wide.Mean.Span[h], 1e-12);
            Assert.True(narrow.Upper.Span[h] - narrow.Lower.Span[h] < wide.Upper.Span[h] - wide.Lower.Span[h]);
            Assert.Equal(wide.Mean.Span[h], narrow.Mean.Span[h]);
        }

        Assert.Empty(wide.Warnings);
        Assert.Throws<ArgumentOutOfRangeException>(() => fit.Forecast(ForecastHorizon.Periods(1), new ForecastOptions { Confidence = 1d }));
    }

    [Fact]
    public void Sum_CarriesTheCovarianceBetweenSteps()
    {
        // Random walk, no drift: the error of the H-step sum is sum_k (H - k + 1) e_k,
        // so its variance is sigma^2 H(H+1)(2H+1)/6 — far above the sum of per-step variances.
        var series = SeriesGenerator.RandomWalk(2_000, seed: 13);
        var fit = new ArimaModel(new ArimaOptions { Order = new(0, 1, 0), IncludeIntercept = false }).Fit(series);
        var forecast = fit.Forecast(ForecastHorizon.Periods(12));

        const int H = 12;
        var total = forecast.Sum(1, H);

        Assert.Equal(H * series[^1], total.Mean, 1e-8);
        Assert.Equal(Math.Sqrt(fit.InnovationVariance * H * (H + 1) * ((2 * H) + 1) / 6d), total.StandardError, 1e-10);
        Assert.Equal(total.Mean - (forecast.CriticalValue * total.StandardError), total.Lower, 1e-12);

        // A single step's aggregate is that step.
        var single = forecast.Sum(5, 1);
        Assert.Equal(forecast.Mean.Span[4], single.Mean, 1e-12);
        Assert.Equal(forecast.StandardError.Span[4], single.StandardError, 1e-12);

        Assert.Throws<ArgumentOutOfRangeException>(() => forecast.Sum(0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => forecast.Sum(12, 2));
    }

    [Fact]
    public void CalendarHorizon_ResolvesThroughTheModelFrequency()
    {
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0), Frequency = SeriesFrequency.Monthly })
            .Fit(SeriesGenerator.Ar1(1_000, 0.5, seed: 14));

        Assert.Equal(24, fit.Forecast(ForecastHorizon.Years(2)).Steps);

        var undeclared = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).Fit(SeriesGenerator.Ar1(1_000, 0.5, seed: 14));
        Assert.Throws<ForecastHorizonException>(() => undeclared.Forecast(ForecastHorizon.Years(2)));
    }

    [Fact]
    public void NonInvertibleModel_WarnsByDefault_AndCanBeRefused()
    {
        var fit = SyntheticFit(phi: [], theta: [1.5], invertible: false);

        var forecast = fit.Forecast(ForecastHorizon.Periods(3));
        Assert.Single(forecast.Warnings);
        Assert.Contains("not invertible", forecast.Warnings[0], StringComparison.Ordinal);

        Assert.Throws<InvalidOperationException>(
            () => fit.Forecast(ForecastHorizon.Periods(3), new ForecastOptions { RequireInvertible = true }));
    }

    /// <summary>A fit assembled by hand, for exercising paths the estimator will not produce.</summary>
    private static ArimaFit SyntheticFit(double[] phi, double[] theta, bool invertible)
    {
        var options = new ArimaOptions { Order = new(phi.Length, 0, theta.Length) };
        var diagnostics = new ArimaDiagnostics(
            100, phi.Length + theta.Length + 2, 100, -100, 200, 201, 210, 4,
            isStationary: true, stationarityMargin: 1, isInvertible: invertible, invertibilityMargin: invertible ? 0.5 : -0.5,
            isConstantSeries: false, new SolveDiagnostics { Succeeded = true, FailedColumn = -1 },
            0, double.NaN, 0, double.NaN, []);
        var seed = new ForecastSeed(
            IntegrationState.FromEnd([1d, 2d, 3d], DifferenceSpec.None),
            [],
            new double[phi.Length],
            new double[theta.Length]);

        return new ArimaFit(options, phi, theta, [], 0d, 1d, 100, diagnostics, new FitWindow(0, 100), [], seed);
    }
}
