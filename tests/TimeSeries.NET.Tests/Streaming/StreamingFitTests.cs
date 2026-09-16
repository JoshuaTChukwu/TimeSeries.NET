using TimeSeries.Data;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Streaming;

public class StreamingFitTests
{
    private static readonly ArimaOptions Options = new() { Order = new(1, 1, 1), PilotOrder = 6 };

    [Theory]
    [InlineData(1)]
    [InlineData(100)]
    [InlineData(8192)]
    [InlineData(10_000)]
    public async Task StreamingFit_EqualsInMemoryFit_Exactly(int batchSize)
    {
        // The M6 acceptance criterion — asked for at 1e-10 relative, delivered bit for bit,
        // because the scan is invariant to batch boundaries.
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(5_000, [0.5], [0.3], seed: 1), 1);
        var model = new ArimaModel(Options);

        var inMemory = model.Fit(series);
        var streamed = await model.FitAsync(new ArraySource(series, batchSize: batchSize));

        AssertSameFit(inMemory, streamed);
    }

    [Fact]
    public async Task StreamingArimaxFit_EqualsInMemoryFit_Exactly()
    {
        var x = ExogenousMatrix.FromColumns(SeriesGenerator.Ar1(4_000, 0.4, seed: 2), SeriesGenerator.Gaussian(4_000, seed: 3));
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(4_000, [0.5], [0.3], seed: 4), 1).Take(4_000).ToArray();
        var model = new ArimaModel(Options);

        var inMemory = model.Fit(series, x);
        var streamed = await model.FitAsync(new ArraySource(series, x, batchSize: 777));

        AssertSameFit(inMemory, streamed);
        Assert.Equal(inMemory.ExogenousCoefficients.ToArray(), streamed.ExogenousCoefficients.ToArray());
    }

    [Fact]
    public async Task AsyncEnumerableFit_EqualsInMemoryFit_Exactly()
    {
        var series = SeriesGenerator.Arma(9_000, [0.6], [0.2], seed: 5);
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1), PilotOrder = 5 });

        static async IAsyncEnumerable<double> Stream(double[] values)
        {
            foreach (var value in values)
            {
                await Task.Yield();
                yield return value;
            }
        }

        AssertSameFit(model.Fit(series), await model.FitAsync(Stream(series)));
    }

    [Fact]
    public async Task TheSourceIsReadExactlyOnce_AndDisposedBeforeTheFitReturns()
    {
        var series = SeriesGenerator.Arma(3_000, [0.5], [], seed: 6);
        var counting = new CountingSource(new ArraySource(series, batchSize: 1000));

        var fit = await new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitAsync(counting);

        Assert.Equal(1, counting.Opens);
        Assert.Equal(1, counting.Disposes);
        Assert.Equal(4, counting.Reads); // three batches and the terminating false
        Assert.Equal(3_000, counting.RowsDelivered);
        Assert.Equal(3_000, fit.ObservationCount);
        Assert.Equal(new FitWindow(0, 3_000), fit.Window);
    }

    [Fact]
    public async Task Cancellation_StopsWithinOneBatch_AndDisposesTheCursor()
    {
        var series = SeriesGenerator.Arma(50_000, [0.5], [], seed: 7);
        using var cts = new CancellationTokenSource();
        var counting = new CountingSource(new ArraySource(series, batchSize: 1000), cts, cancelAfter: 5);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitAsync(counting, cts.Token).AsTask());

        Assert.Equal(1, counting.Disposes);
        Assert.True(counting.RowsDelivered <= 6_000, $"Read {counting.RowsDelivered} rows after cancellation at read 6.");
    }

    [Fact]
    public async Task FitMany_FitsEachKey_AndReportsFailuresWithoutAborting()
    {
        var a = SeriesGenerator.Ar1(3_000, 0.3, seed: 8);
        var b = SeriesGenerator.Ar1(3_000, 0.8, seed: 9);
        var c = SeriesGenerator.Ar1(3_000, -0.5, seed: 10);
        var tooShort = SeriesGenerator.Ar1(20, 0.5, seed: 11);
        var poisoned = SeriesGenerator.Ar1(3_000, 0.5, seed: 12);
        poisoned[1_234] = double.NaN;

        var source = new InMemoryGroupedSource(batchSize: 500)
            .Add("acct-a", a)
            .Add("acct-short", tooShort)
            .Add("acct-b", b)
            .Add("acct-nan", poisoned)
            .Add("acct-c", c);
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) });

        var results = new List<KeyedArimaFit>();
        await foreach (var result in model.FitManyAsync(source, FitManyOptions.Sequential))
        {
            results.Add(result);
        }

        Assert.Equal(["acct-a", "acct-short", "acct-b", "acct-nan", "acct-c"], results.Select(r => r.Key).ToArray());

        AssertSameFit(model.Fit(a), results[0].Fit!);
        AssertSameFit(model.Fit(b), results[2].Fit!);
        AssertSameFit(model.Fit(c), results[4].Fit!);

        Assert.False(results[1].Succeeded);
        Assert.IsType<InsufficientDataException>(results[1].Error);
        Assert.Equal(20, results[1].ObservationCount);

        Assert.False(results[3].Succeeded);
        var invalid = Assert.IsType<InvalidSeriesException>(results[3].Error);
        Assert.Equal(1_234, invalid.Index);
    }

    [Fact]
    public async Task NullSources_AreRejected()
    {
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) });

        await Assert.ThrowsAsync<ArgumentNullException>(() => model.FitAsync((ITimeSeriesSource)null!).AsTask());
        await Assert.ThrowsAsync<ArgumentNullException>(() => model.FitAsync((IAsyncEnumerable<double>)null!).AsTask());
        await Assert.ThrowsAsync<ArgumentNullException>(async () =>
        {
            await foreach (var _ in model.FitManyAsync(null!))
            {
            }
        });
    }

    [Fact]
    public async Task ArraySource_SatisfiesTheContract()
    {
        var values = SeriesGenerator.Uniform(2_345, seed: 13);
        var exogenous = ExogenousMatrix.FromColumns(SeriesGenerator.Uniform(2_345, seed: 14), SeriesGenerator.Uniform(2_345, seed: 15));

        await SourceContract.VerifyAsync(new ArraySource(values, batchSize: 500), values, null, 500);
        await SourceContract.VerifyAsync(new ArraySource(values, exogenous, batchSize: 1000), values, exogenous, 1000);
        await SourceContract.VerifyAsync(new ArraySource(Array.Empty<double>(), batchSize: 8), [], null, 8);

        Assert.Throws<ArgumentException>(() => new ArraySource(values, ExogenousMatrix.FromColumn(new double[3])));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ArraySource(values, batchSize: 0));
    }

    [Fact]
    public async Task OneMillionRows_FitInBoundedMemory()
    {
        const long Rows = 1_000_000;
        var source = new SyntheticAr1Source(Rows, phi: 0.6, seed: 16);
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1), PilotOrder = 8 });

        // Warm the JIT and the scan's scratch so the measured pass is steady state.
        await model.FitAsync(new SyntheticAr1Source(20_000, 0.6, seed: 17));

        var before = GC.GetAllocatedBytesForCurrentThread();
        var fit = await model.FitAsync(source);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(Rows, fit.ObservationCount);
        Assert.Equal(0.6, fit.AutoRegressive.Span[0], 0.05);

        // A million rows is 8 MB of data. The fit must allocate a small fixed amount
        // — scratch buffers and the frozen state — not anything that scales with rows.
        Assert.True(allocated < 2 * 1024 * 1024, $"Fit allocated {allocated:N0} bytes for {Rows:N0} rows.");
    }

    [Fact]
    public async Task HundredMillionRows_FitUnderTheMemoryCeiling_WhenEnabled()
    {
        // Roughly a minute of arithmetic; run with TIMESERIES_LONG_TESTS=1.
        if (Environment.GetEnvironmentVariable("TIMESERIES_LONG_TESTS") is null)
        {
            return;
        }

        const long Rows = 100_000_000;
        var model = new ArimaModel(new ArimaOptions { Order = new(2, 1, 1), PilotOrder = 8 });

        GC.Collect();
        var baseline = GC.GetTotalMemory(forceFullCollection: true);
        var fit = await model.FitAsync(new SyntheticAr1Source(Rows, phi: 0.5, seed: 18));
        var after = GC.GetTotalMemory(forceFullCollection: false);

        Assert.Equal(Rows, fit.ObservationCount);
        Assert.True(after - baseline < 64L * 1024 * 1024, $"Managed heap grew by {(after - baseline) / 1024 / 1024} MB.");
    }

    private static void AssertSameFit(ArimaFit expected, ArimaFit actual)
    {
        Assert.Equal(expected.AutoRegressive.ToArray(), actual.AutoRegressive.ToArray());
        Assert.Equal(expected.MovingAverage.ToArray(), actual.MovingAverage.ToArray());
        Assert.Equal(expected.Intercept, actual.Intercept);
        Assert.Equal(expected.InnovationVariance, actual.InnovationVariance);
        Assert.Equal(expected.ObservationCount, actual.ObservationCount);
        Assert.Equal(expected.Diagnostics.Aicc, actual.Diagnostics.Aicc);
        Assert.Equal(expected.Seed.RecentResiduals, actual.Seed.RecentResiduals);
        Assert.Equal(expected.Seed.RecentValues, actual.Seed.RecentValues);
        Assert.Equal(expected.Seed.Integration.Tail.ToArray(), actual.Seed.Integration.Tail.ToArray());
    }
}

public class ExogenousFromSourceTests
{
    [Fact]
    public async Task ReadsRegressorsRowForRow_IgnoringValues()
    {
        var x = ExogenousMatrix.FromColumns(SeriesGenerator.Uniform(1_000, seed: 1), SeriesGenerator.Uniform(1_000, seed: 2));
        var source = new ArraySource(new double[1_000], x, batchSize: 128);

        var read = await ExogenousMatrix.FromSourceAsync(source);

        Assert.Equal(x.Count, read.Count);
        Assert.Equal(x.RegressorCount, read.RegressorCount);
        Assert.Equal(x.Values.ToArray(), read.Values.ToArray());
    }

    [Fact]
    public async Task RequiresRegressors()
    {
        await Assert.ThrowsAsync<ArgumentException>(() => ExogenousMatrix.FromSourceAsync(new ArraySource(new double[10])).AsTask());
        await Assert.ThrowsAsync<ArgumentNullException>(() => ExogenousMatrix.FromSourceAsync(null!).AsTask());
    }
}
