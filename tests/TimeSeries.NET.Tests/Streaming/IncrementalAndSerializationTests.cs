using TimeSeries.Data;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Streaming;

public class IncrementalArimaTests
{
    private static readonly ArimaOptions Options = new() { Order = new(1, 1, 1), PilotOrder = 6 };

    [Fact]
    public void FoldInPieces_EqualsBatchOverTheUnion_Exactly()
    {
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(6_000, [0.5], [0.3], seed: 1), 1);
        var batch = new ArimaModel(Options).Fit(series);

        var incremental = new IncrementalArima(Options);
        incremental.Fold(series.AsSpan(0, 1_000));
        incremental.Fold(series.AsSpan(1_000, 2_500));
        incremental.Fold(series.AsSpan(3_500));

        Assert.Equal(3, incremental.FoldCount);
        Assert.Equal(new FitWindow(0, series.Length), incremental.Watermark);
        AssertSameFit(batch, incremental.Solve());
    }

    [Fact]
    public async Task FoldAsyncFromSources_EqualsBatch_Exactly()
    {
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(6_000, [0.5], [0.3], seed: 2), 1);
        var batch = new ArimaModel(Options).Fit(series);

        var incremental = new IncrementalArima(Options);
        await incremental.FoldAsync(new ArraySource(series.AsMemory(0, 2_000), batchSize: 300));
        await incremental.FoldAsync(new ArraySource(series.AsMemory(2_000), batchSize: 8192));

        AssertSameFit(batch, incremental.Solve());
    }

    [Fact]
    public void FoldWithRegressors_EqualsBatch_Exactly()
    {
        var x = ExogenousMatrix.FromColumn(SeriesGenerator.Ar1(5_001, 0.4, seed: 3));
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(5_000, [0.5], [0.3], seed: 4), 1);
        var batch = new ArimaModel(Options).Fit(series, x);

        var incremental = new IncrementalArima(Options, regressorCount: 1);
        incremental.Fold(series.AsSpan(0, 3_000), new ExogenousMatrix(x.Values.Slice(0, 3_000), 1));
        incremental.Fold(series.AsSpan(3_000), new ExogenousMatrix(x.Values.Slice(3_000), 1));

        var fit = incremental.Solve();
        AssertSameFit(batch, fit);
        Assert.Equal(batch.ExogenousCoefficients.ToArray(), fit.ExogenousCoefficients.ToArray());
    }

    [Fact]
    public void SaveThenRestore_ThenFoldTheRest_EqualsBatch_Exactly()
    {
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(6_000, [0.5], [0.3], seed: 5), 1);
        var batch = new ArimaModel(Options).Fit(series);

        var first = new IncrementalArima(Options);
        first.Fold(series.AsSpan(0, 2_345));

        using var state = new MemoryStream();
        first.SaveTo(state);
        state.Position = 0;

        // The state is a few kilobytes: (D+1)^2 + D + 2 doubles plus windows and tail.
        Assert.True(state.Length < 24 * 1024, $"State is {state.Length} bytes.");

        var restored = IncrementalArima.Restore(state, Options);
        Assert.Equal(1, restored.FoldCount);
        Assert.Equal(new FitWindow(0, 2_345), restored.Watermark);

        restored.Fold(series.AsSpan(2_345));
        AssertSameFit(batch, restored.Solve());
    }

    [Fact]
    public void SaveBeforeWarmup_RestoresMidTransform()
    {
        // Two rows in: the differencing has not even warmed up. The state must still round-trip.
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(3_000, [0.5], [0.3], seed: 6), 1);
        var batch = new ArimaModel(Options).Fit(series);

        var first = new IncrementalArima(Options);
        first.Fold(series.AsSpan(0, 2));

        using var state = new MemoryStream();
        first.SaveTo(state);
        state.Position = 0;

        var restored = IncrementalArima.Restore(state, Options);
        restored.Fold(series.AsSpan(2));

        AssertSameFit(batch, restored.Solve());
    }

    [Fact]
    public void Restore_WithADifferentModel_IsRefused()
    {
        var incremental = new IncrementalArima(Options);
        incremental.Fold(SeriesGenerator.Uniform(100, seed: 7));

        using var state = new MemoryStream();
        incremental.SaveTo(state);
        state.Position = 0;

        Assert.Throws<InvalidDataException>(() => IncrementalArima.Restore(state, new ArimaOptions { Order = new(2, 1, 1), PilotOrder = 6 }));

        using var garbage = new MemoryStream([1, 2, 3, 4, 5, 6, 7, 8]);
        Assert.Throws<InvalidDataException>(() => IncrementalArima.Restore(garbage, Options));
    }

    [Fact]
    public void ForgettingFactor_ScalesEarlierFoldsBeforeEachNewOne()
    {
        var options = new ArimaOptions { Order = new(1, 0, 0), ForgettingFactor = 0.5 };
        var series = SeriesGenerator.Ar1(4_000, 0.5, seed: 8);

        var incremental = new IncrementalArima(options);
        incremental.Fold(series.AsSpan(0, 2_000));
        var first = incremental.Solve().Diagnostics.EffectiveObservations;

        incremental.Fold(series.AsSpan(2_000));
        var second = incremental.Solve().Diagnostics.EffectiveObservations;

        // AR(1) with the default ten Ljung-Box lags: rows complete once eleven values are
        // in, so fold one contributes 1989 rows, halved before fold two adds its 2000.
        Assert.Equal(1_989, first);
        Assert.Equal((0.5 * first) + 2_000, second);
    }

    [Fact]
    public void SolveCanBeRepeated_WithoutChangingTheState()
    {
        var incremental = new IncrementalArima(Options);
        incremental.Fold(SeriesGenerator.Integrate(SeriesGenerator.Arma(2_000, [0.5], [0.3], seed: 9), 1));

        var a = incremental.Solve();
        var b = incremental.Solve();

        AssertSameFit(a, b);
    }

    [Fact]
    public void ArgumentValidation()
    {
        Assert.Throws<ArgumentNullException>(() => new IncrementalArima(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new IncrementalArima(Options, regressorCount: -1));

        var withRegressors = new IncrementalArima(Options, regressorCount: 1);
        var threw = false;
        try
        {
            withRegressors.Fold(new double[10]);
        }
        catch (InvalidOperationException)
        {
            threw = true;
        }

        Assert.True(threw);
        Assert.Throws<ArgumentNullException>(() => new IncrementalArima(Options).SaveTo(null!));
    }

    private static void AssertSameFit(ArimaFit expected, ArimaFit actual)
    {
        Assert.Equal(expected.AutoRegressive.ToArray(), actual.AutoRegressive.ToArray());
        Assert.Equal(expected.MovingAverage.ToArray(), actual.MovingAverage.ToArray());
        Assert.Equal(expected.Intercept, actual.Intercept);
        Assert.Equal(expected.InnovationVariance, actual.InnovationVariance);
        Assert.Equal(expected.ObservationCount, actual.ObservationCount);
        Assert.Equal(expected.Diagnostics.EffectiveObservations, actual.Diagnostics.EffectiveObservations);
        Assert.Equal(expected.Seed.RecentResiduals, actual.Seed.RecentResiduals);
        Assert.Equal(expected.Seed.RecentValues, actual.Seed.RecentValues);
        Assert.Equal(expected.Seed.Integration.Tail.ToArray(), actual.Seed.Integration.Tail.ToArray());
    }
}

public class ArimaFitSerializationTests
{
    [Fact]
    public void WriteThenRead_ReproducesTheFitAndItsForecasts()
    {
        var x = ExogenousMatrix.FromColumns(SeriesGenerator.Ar1(4_000, 0.4, seed: 1), SeriesGenerator.Gaussian(4_000, seed: 2));
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(4_000, [0.5, 0.2], [0.3], seed: 3), 1).Take(4_000).ToArray();
        var options = new ArimaOptions
        {
            Order = new(2, 1, 1),
            Seasonal = new(1, 4),
            PilotOrder = 7,
            Ridge = 1e-9,
            Frequency = SeriesFrequency.Quarterly,
        };
        var original = new ArimaModel(options).Fit(series, x);

        using var stream = new MemoryStream();
        original.WriteTo(stream);
        stream.Position = 0;
        var restored = ArimaFit.ReadFrom(stream);

        Assert.Equal(original.Options, restored.Options);
        Assert.Equal(original.AutoRegressive.ToArray(), restored.AutoRegressive.ToArray());
        Assert.Equal(original.MovingAverage.ToArray(), restored.MovingAverage.ToArray());
        Assert.Equal(original.ExogenousCoefficients.ToArray(), restored.ExogenousCoefficients.ToArray());
        Assert.Equal(original.Intercept, restored.Intercept);
        Assert.Equal(original.InnovationVariance, restored.InnovationVariance);
        Assert.Equal(original.ObservationCount, restored.ObservationCount);
        Assert.Equal(original.Window, restored.Window);
        Assert.Equal(original.StandardErrors.ToArray(), restored.StandardErrors.ToArray());
        Assert.Equal(original.Diagnostics.Aicc, restored.Diagnostics.Aicc);
        Assert.Equal(original.Diagnostics.PilotOrder, restored.Diagnostics.PilotOrder);
        Assert.Equal(original.Diagnostics.IsInvertible, restored.Diagnostics.IsInvertible);
        Assert.Equal(original.Diagnostics.Solve, restored.Diagnostics.Solve);
        Assert.Equal(original.Diagnostics.LjungBoxStatistic, restored.Diagnostics.LjungBoxStatistic);
        Assert.Equal(original.Diagnostics.LjungBoxPValue, restored.Diagnostics.LjungBoxPValue);
        Assert.Equal(original.Diagnostics.ResidualAutocorrelations.ToArray(), restored.Diagnostics.ResidualAutocorrelations.ToArray());

        var future = ExogenousMatrix.FromColumns(SeriesGenerator.Ar1(8, 0.4, seed: 4), SeriesGenerator.Gaussian(8, seed: 5));
        var a = original.Forecast(ForecastHorizon.Years(2), future);
        var b = restored.Forecast(ForecastHorizon.Years(2), future);

        Assert.Equal(a.Mean.ToArray(), b.Mean.ToArray());
        Assert.Equal(a.StandardError.ToArray(), b.StandardError.ToArray());
    }

    [Fact]
    public void AnArima211Fit_IsAFewHundredBytes()
    {
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(3_000, [0.5, 0.2], [0.3], seed: 6), 1);
        var fit = new ArimaModel(new ArimaOptions { Order = new(2, 1, 1) }).Fit(series);

        using var stream = new MemoryStream();
        fit.WriteTo(stream);

        Assert.True(stream.Length < 512, $"Serialised fit is {stream.Length} bytes.");
    }

    [Fact]
    public void GarbageIsRefused()
    {
        using var garbage = new MemoryStream(new byte[64]);
        Assert.Throws<InvalidDataException>(() => ArimaFit.ReadFrom(garbage));
        Assert.Throws<ArgumentNullException>(() => ArimaFit.ReadFrom(null!));

        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).Fit(SeriesGenerator.Ar1(500, 0.5, seed: 7));
        Assert.Throws<ArgumentNullException>(() => fit.WriteTo(null!));
    }
}
