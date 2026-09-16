using TimeSeries.Data;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Streaming;

public class ParallelFitManyTests
{
    private static readonly ArimaModel Model = new(new ArimaOptions { Order = new(1, 0, 0) });

    private static (InMemoryGroupedSource Source, Dictionary<string, double[]> Series) Population(int keys, int seed, int batchSize = 256)
    {
        var source = new InMemoryGroupedSource(batchSize: batchSize);
        var series = new Dictionary<string, double[]>(StringComparer.Ordinal);

        for (var i = 0; i < keys; i++)
        {
            var key = $"ACC-{i:D4}";
            var phi = -0.8 + (1.6 * i / Math.Max(1, keys - 1));
            var values = SeriesGenerator.Ar1(1_500 + (i * 7), phi, seed + i);
            series[key] = values;
            source.Add(key, values);
        }

        return (source, series);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public async Task Workers_ProduceTheSameFitsAsTheSequentialPath(int workers)
    {
        var (source, series) = Population(40, seed: 100);
        var options = new FitManyOptions { DegreeOfParallelism = workers, BatchesInFlightPerKey = 2 };

        var results = new List<KeyedArimaFit>();
        await foreach (var result in Model.FitManyAsync(source, options))
        {
            results.Add(result);
        }

        Assert.Equal(40, results.Count);
        Assert.Equal(series.Keys.OrderBy(k => k, StringComparer.Ordinal), results.Select(r => r.Key).OrderBy(k => k, StringComparer.Ordinal));

        foreach (var result in results)
        {
            Assert.True(result.Succeeded, $"{result.Key}: {result.Error?.Message}");
            var direct = Model.Fit(series[result.Key]);
            Assert.Equal(direct.AutoRegressive.ToArray(), result.Fit!.AutoRegressive.ToArray());
            Assert.Equal(direct.Intercept, result.Fit.Intercept);
            Assert.Equal(series[result.Key].Length, result.ObservationCount);
        }
    }

    [Fact]
    public async Task Failures_AreReportedPerKey_WithoutAbortingTheScan()
    {
        var good = SeriesGenerator.Ar1(2_000, 0.5, seed: 1);
        var poisoned = SeriesGenerator.Ar1(2_000, 0.5, seed: 2);
        poisoned[777] = double.NaN;
        var source = new InMemoryGroupedSource(batchSize: 100)
            .Add("a", good)
            .Add("b", SeriesGenerator.Ar1(15, 0.5, seed: 3))
            .Add("c", poisoned)
            .Add("d", good);

        var results = new Dictionary<string, KeyedArimaFit>(StringComparer.Ordinal);
        await foreach (var result in Model.FitManyAsync(source, new FitManyOptions { DegreeOfParallelism = 3 }))
        {
            results[result.Key] = result;
        }

        Assert.Equal(4, results.Count);
        Assert.True(results["a"].Succeeded);
        Assert.True(results["d"].Succeeded);
        Assert.IsType<InsufficientDataException>(results["b"].Error);
        Assert.Equal(777, Assert.IsType<InvalidSeriesException>(results["c"].Error).Index);
    }

    [Fact]
    public async Task SourceFailure_Propagates()
    {
        var (inner, _) = Population(10, seed: 200);
        var source = new FailingGroupedSource(inner, failAfterReads: 12);

        await Assert.ThrowsAsync<SeriesOrderException>(async () =>
        {
            await foreach (var _ in Model.FitManyAsync(source, new FitManyOptions { DegreeOfParallelism = 4 }))
            {
            }
        });

        Assert.True(source.Disposed);
    }

    [Fact]
    public async Task Cancellation_StopsTheScan()
    {
        var (source, _) = Population(30, seed: 300, batchSize: 64);
        using var cts = new CancellationTokenSource();
        var seen = 0;

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in Model.FitManyAsync(source, new FitManyOptions { DegreeOfParallelism = 4 }, cts.Token))
            {
                if (++seen == 3)
                {
                    cts.Cancel();
                }
            }
        });

        Assert.True(seen >= 3);
    }

    [Fact]
    public async Task SequentialOptions_UseTheOrderedPath()
    {
        var (source, series) = Population(5, seed: 400);

        var keys = new List<string>();
        await foreach (var result in Model.FitManyAsync(source, FitManyOptions.Sequential))
        {
            keys.Add(result.Key);
            Assert.Equal(Model.Fit(series[result.Key]).AutoRegressive.ToArray(), result.Fit!.AutoRegressive.ToArray());
        }

        Assert.Equal(series.Keys.OrderBy(k => k, StringComparer.Ordinal), keys);
    }

    [Fact]
    public void OptionsValidation()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FitManyOptions { DegreeOfParallelism = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new FitManyOptions { BatchesInFlightPerKey = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new FitManyOptions { KeysInFlight = -1 }.Validate());
        Assert.Equal(8, new FitManyOptions { DegreeOfParallelism = 4 }.EffectiveKeysInFlight);
        Assert.Throws<ArgumentNullException>(() => Model.FitManyAsync(new InMemoryGroupedSource(), null!));
    }

    /// <summary>A grouped source that throws an ordering fault part-way, as a database adapter would.</summary>
    private sealed class FailingGroupedSource(IGroupedTimeSeriesSource inner, int failAfterReads) : IGroupedTimeSeriesSource
    {
        public bool Disposed { get; private set; }

        public int RegressorCount => inner.RegressorCount;

        public async ValueTask<IGroupedTimeSeriesCursor> OpenAsync(CancellationToken cancellationToken = default)
            => new Cursor(this, await inner.OpenAsync(cancellationToken), failAfterReads);

        private sealed class Cursor(FailingGroupedSource owner, IGroupedTimeSeriesCursor inner, int failAfterReads) : IGroupedTimeSeriesCursor
        {
            private int _reads;

            public KeyedSeriesBatch Current => inner.Current;

            public ValueTask<bool> ReadAsync(CancellationToken cancellationToken = default)
            {
                if (++_reads > failAfterReads)
                {
                    throw new SeriesOrderException("ACC-0002", "ACC-0001", "Simulated ordering fault.");
                }

                return inner.ReadAsync(cancellationToken);
            }

            public ValueTask DisposeAsync()
            {
                owner.Disposed = true;
                return inner.DisposeAsync();
            }
        }
    }
}
