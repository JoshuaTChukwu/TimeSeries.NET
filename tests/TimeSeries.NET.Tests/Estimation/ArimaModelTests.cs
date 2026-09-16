using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Estimation;

public class ArimaModelTests
{
    [Fact]
    public void ConstantDifferencedSeries_IsShortCircuited()
    {
        // y = 3 + 2t: first differences are exactly 2 everywhere.
        var series = Enumerable.Range(0, 200).Select(t => 3d + (2d * t)).ToArray();

        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 1, 1) }).Fit(series);

        Assert.True(fit.Diagnostics.IsConstantSeries);
        Assert.Equal(2d, fit.Intercept, 1e-12);
        Assert.Equal(0d, fit.InnovationVariance);
        Assert.Equal(0d, fit.AutoRegressive.Span[0]);
        Assert.Equal(0d, fit.MovingAverage.Span[0]);
        Assert.True(double.IsNaN(fit.Diagnostics.Aicc));
        Assert.All(fit.StandardErrors.ToArray(), se => Assert.True(double.IsNaN(se)));
    }

    [Fact]
    public void TooShortASeries_ThrowsWithTheNumbers()
    {
        // Default MaxPilotOrder 32 with q = 1 gives L = 33; 30 observations cannot fit it.
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1) });
        var series = SeriesGenerator.Uniform(30, seed: 1);

        var exception = Assert.Throws<InsufficientDataException>(() => model.Fit(series));

        Assert.Equal(30, exception.Available);
        Assert.Equal(33 + (10 * 3), exception.Required);
        Assert.Contains("MaxPilotOrder", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EmptySeries_ThrowsInsufficientData()
    {
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) });

        Assert.Throws<InsufficientDataException>(() => model.Fit([]));
    }

    [Fact]
    public void NaNInTheSeries_ThrowsNamingTheRow()
    {
        var series = SeriesGenerator.Uniform(500, seed: 1);
        series[137] = double.NaN;

        var exception = Assert.Throws<InvalidSeriesException>(
            () => new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).Fit(series));

        Assert.Equal(137, exception.Index);
    }

    [Fact]
    public void NaNInARegressor_ThrowsNamingTheFlatIndex()
    {
        var series = SeriesGenerator.Uniform(500, seed: 1);
        var x = SeriesGenerator.Uniform(500, seed: 2);
        x[42] = double.PositiveInfinity;

        var exception = Assert.Throws<InvalidSeriesException>(
            () => new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).Fit(series, ExogenousMatrix.FromColumn(x)));

        Assert.Equal(42, exception.Index);
    }

    [Fact]
    public void MisalignedRegressors_Throw()
    {
        var series = SeriesGenerator.Uniform(500, seed: 1);
        var x = SeriesGenerator.Uniform(499, seed: 2);
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) });

        Assert.Throws<ArgumentException>(() => model.Fit(series, ExogenousMatrix.FromColumn(x)));
        Assert.Throws<ArgumentNullException>(() => model.Fit(series, null!));
    }

    [Theory]
    [InlineData(-1, 0, 0)]
    [InlineData(0, -1, 0)]
    [InlineData(0, 0, -1)]
    public void NegativeOrders_AreRejected(int p, int d, int q)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ArimaModel(new ArimaOptions { Order = new(p, d, q) }));
    }

    [Fact]
    public void InconsistentOptions_AreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ArimaModel(new ArimaOptions { Order = new(1, 0, 1), PilotOrder = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ArimaModel(new ArimaOptions { Order = new(2, 0, 2), MaxPilotOrder = 3 }));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ArimaModel(new ArimaOptions { ForgettingFactor = 0d }));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ArimaModel(new ArimaOptions { ForgettingFactor = 1.1 }));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ArimaModel(new ArimaOptions { Ridge = -1e-6 }));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ArimaModel(new ArimaOptions { Seasonal = new(1, 1) }));
        Assert.Throws<ArgumentNullException>(() => new ArimaModel(null!));
    }

    [Fact]
    public void PilotOrder_IsZeroForPureAr_AndAtLeastPPlusQOtherwise()
    {
        var series = SeriesGenerator.Arma(3_000, [0.5], [0.3], seed: 3);

        var ar = new ArimaModel(new ArimaOptions { Order = new(2, 0, 0) }).Fit(series);
        var arma = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1) }).Fit(series);
        var fixedPilot = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1), PilotOrder = 7 }).Fit(series);

        Assert.Equal(0, ar.Diagnostics.PilotOrder);
        Assert.True(arma.Diagnostics.PilotOrder >= 2);
        Assert.Equal(7, fixedPilot.Diagnostics.PilotOrder);
    }

    [Fact]
    public void Diagnostics_AreFiniteAndConsistent()
    {
        var series = SeriesGenerator.Arma(2_000, [0.5], [0.3], seed: 4);
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1), PilotOrder = 6 }).Fit(series);
        var d = fit.Diagnostics;

        // Effective rows: differenced count less the lag depth max(p, q + m) = 7.
        Assert.Equal(2_000 - 7, d.EffectiveObservations);
        Assert.Equal(4, d.ParameterCount); // intercept, phi, theta, variance
        Assert.True(d.Aic < d.Aicc);
        Assert.True(d.Aicc < d.Bic);
        Assert.True(double.IsFinite(d.LogLikelihood));
        Assert.Equal(d.ResidualSumOfSquares / d.EffectiveObservations, fit.InnovationVariance, 1e-12);
        Assert.True(d.Solve.Succeeded);
        Assert.True(d.Solve.PivotRatio < 1e6);
    }

    [Fact]
    public void WithoutAnIntercept_TheConstantIsZeroAndNothingIsShifted()
    {
        var series = SeriesGenerator.Arma(4_000, [0.6], [], seed: 5);
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0), IncludeIntercept = false }).Fit(series);

        Assert.Equal(0d, fit.Intercept);
        Assert.Single(fit.StandardErrors.ToArray());
        Assert.Equal(0.6, fit.AutoRegressive.Span[0], 0.03);
    }

    [Fact]
    public void SeasonalDifferencing_FitsOnTheSeasonallyDifferencedScale()
    {
        // A random walk plus a fixed 12-period pattern: (1-B)(1-B^12) removes both.
        const int Period = 12;
        var differenced = SeriesGenerator.Arma(3_000, [0.4], [], seed: 8);
        var series = SeriesGenerator.Integrate(differenced, 1);
        for (var t = 0; t < series.Length; t++)
        {
            series[t] += 5d * Math.Sin(2d * Math.PI * t / Period);
        }

        var fit = new ArimaModel(new ArimaOptions
        {
            Order = new(1, 1, 0),
            Seasonal = new(1, Period),
        }).Fit(series);

        Assert.Equal(new SeasonalOrder(1, Period), fit.Seasonal);
        Assert.Equal(1 + Period, fit.Seed.Integration.Tail.Length);
        Assert.True(fit.Diagnostics.IsStationary);
    }

    [Fact]
    public void TheFitHoldsNothingProportionalToTheSeries()
    {
        var series = SeriesGenerator.Arma(50_000, [0.5, 0.2], [0.3], seed: 6);
        var fit = new ArimaModel(new ArimaOptions { Order = new(2, 0, 1) }).Fit(series);

        Assert.Equal(2, fit.Seed.RecentValues.Length);
        Assert.Single(fit.Seed.RecentResiduals);
        Assert.Equal(0, fit.Seed.Integration.Tail.Length);
        Assert.Equal(50_000, fit.ObservationCount);
    }

    [Fact]
    public void RecentResiduals_HaveTheInnovationScale()
    {
        // Innovations of unit variance: the last reconstructed residual should look like one.
        var series = SeriesGenerator.Arma(5_000, [0.5], [0.4], seed: 12);
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1) }).Fit(series);

        Assert.True(Math.Abs(fit.Seed.RecentResiduals[0]) < 5d);
        Assert.Equal(series[^1], fit.Seed.RecentValues[0]);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(100)]
    [InlineData(8192)]
    public void Scan_IsInvariantToBatchBoundaries(int batchSize)
    {
        // The accumulators see exactly the same sequence of rows whatever the batch size,
        // so the frozen Gram must be bit-for-bit identical — the property the streaming
        // path relies on.
        var series = SeriesGenerator.Arma(3_000, [0.5], [0.3], seed: 21);
        var x = SeriesGenerator.Ar1(3_000, 0.5, seed: 22);
        var options = new ArimaOptions { Order = new(1, 1, 1), PilotOrder = 4 };

        var whole = new ArimaScan(options, regressorCount: 1);
        whole.Accept(series, x);

        var batched = new ArimaScan(options, regressorCount: 1);
        for (var start = 0; start < series.Length; start += batchSize)
        {
            var take = Math.Min(batchSize, series.Length - start);
            batched.Accept(series.AsSpan(start, take), x.AsSpan(start, take));
        }

        Assert.Equal(whole.DifferencedCount, batched.DifferencedCount);
        Assert.Equal(whole.Gram.Freeze().Values.ToArray(), batched.Gram.Freeze().Values.ToArray());
        Assert.Equal(whole.Gram.Freeze().Sums.ToArray(), batched.Gram.Freeze().Sums.ToArray());

        var a = whole.Solve(new FitWindow(0, series.Length));
        var b = batched.Solve(new FitWindow(0, series.Length));
        Assert.Equal(a.AutoRegressive.ToArray(), b.AutoRegressive.ToArray());
        Assert.Equal(a.MovingAverage.ToArray(), b.MovingAverage.ToArray());
        Assert.Equal(a.ExogenousCoefficients.ToArray(), b.ExogenousCoefficients.ToArray());
    }

    [Fact]
    public void ExogenousMatrix_Validation()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ExogenousMatrix(new double[4], 0));
        Assert.Throws<ArgumentException>(() => new ExogenousMatrix(new double[5], 2));
        Assert.Throws<ArgumentException>(() => ExogenousMatrix.FromColumns([1, 2], [1, 2, 3]));
        Assert.Throws<ArgumentException>(() => ExogenousMatrix.FromColumns());

        var matrix = ExogenousMatrix.FromColumns([1, 2, 3], [10, 20, 30]);
        Assert.Equal(3, matrix.Count);
        Assert.Equal(2, matrix.RegressorCount);
        Assert.Equal(20d, matrix[1, 1]);
        Assert.Equal([3d, 30d], matrix.Row(2).ToArray());
        Assert.Equal([2d, 20d, 3d, 30d], matrix.Rows(1, 2).ToArray());
    }

    [Fact]
    public void ModelIsReusableAcrossFits()
    {
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) });
        var a = model.Fit(SeriesGenerator.Ar1(2_000, 0.3, seed: 1));
        var b = model.Fit(SeriesGenerator.Ar1(2_000, 0.8, seed: 2));

        Assert.Equal(0.3, a.AutoRegressive.Span[0], 0.05);
        Assert.Equal(0.8, b.AutoRegressive.Span[0], 0.05);
    }
}
