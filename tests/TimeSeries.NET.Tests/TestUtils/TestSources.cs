using TimeSeries.Data;

namespace TimeSeries.Tests.TestUtils;

/// <summary>
/// Wraps a source and counts what the estimator does with it. The one-pass claim is
/// asserted through this: exactly one open, every batch read, exactly one dispose.
/// </summary>
public sealed class CountingSource : ITimeSeriesSource
{
    private readonly ITimeSeriesSource _inner;
    private readonly int _cancelAfterReads;
    private readonly CancellationTokenSource? _cts;

    public CountingSource(ITimeSeriesSource inner, CancellationTokenSource? cancelAfterReads = null, int cancelAfter = int.MaxValue)
    {
        _inner = inner;
        _cts = cancelAfterReads;
        _cancelAfterReads = cancelAfter;
    }

    public int Opens { get; private set; }

    public int Reads { get; private set; }

    public int Disposes { get; private set; }

    public long RowsDelivered { get; private set; }

    public int RegressorCount => _inner.RegressorCount;

    public async ValueTask<ITimeSeriesCursor> OpenAsync(CancellationToken cancellationToken = default)
    {
        Opens++;
        return new Cursor(this, await _inner.OpenAsync(cancellationToken));
    }

    private sealed class Cursor(CountingSource owner, ITimeSeriesCursor inner) : ITimeSeriesCursor
    {
        public SeriesBatch Current => inner.Current;

        public async ValueTask<bool> ReadAsync(CancellationToken cancellationToken = default)
        {
            owner.Reads++;

            if (owner.Reads > owner._cancelAfterReads)
            {
                owner._cts?.Cancel();
            }

            var more = await inner.ReadAsync(cancellationToken);
            if (more)
            {
                owner.RowsDelivered += inner.Current.Count;
            }

            return more;
        }

        public ValueTask DisposeAsync()
        {
            owner.Disposes++;
            return inner.DisposeAsync();
        }
    }
}

/// <summary>A keyed population held in memory, delivered key by key in insertion order.</summary>
public sealed class InMemoryGroupedSource : IGroupedTimeSeriesSource
{
    private readonly List<(string Key, double[] Values, ExogenousMatrix? Exogenous)> _series = [];
    private readonly int _batchSize;

    public InMemoryGroupedSource(int regressorCount = 0, int batchSize = 8192)
    {
        RegressorCount = regressorCount;
        _batchSize = batchSize;
    }

    public int RegressorCount { get; }

    public InMemoryGroupedSource Add(string key, double[] values, ExogenousMatrix? exogenous = null)
    {
        _series.Add((key, values, exogenous));
        return this;
    }

    public ValueTask<IGroupedTimeSeriesCursor> OpenAsync(CancellationToken cancellationToken = default)
        => new(new Cursor(this));

    private sealed class Cursor(InMemoryGroupedSource owner) : IGroupedTimeSeriesCursor
    {
        private int _series = -1;
        private ITimeSeriesCursor? _inner;

        public KeyedSeriesBatch Current { get; private set; }

        public async ValueTask<bool> ReadAsync(CancellationToken cancellationToken = default)
        {
            while (true)
            {
                if (_inner is not null && await _inner.ReadAsync(cancellationToken))
                {
                    Current = new KeyedSeriesBatch(owner._series[_series].Key, _inner.Current);
                    return true;
                }

                _series++;
                if (_series >= owner._series.Count)
                {
                    return false;
                }

                var (_, values, exogenous) = owner._series[_series];
                _inner = await new ArraySource(values, exogenous, owner._batchSize).OpenAsync(cancellationToken);
            }
        }

        public ValueTask DisposeAsync() => _inner?.DisposeAsync() ?? default;
    }
}

/// <summary>
/// An AR(1) generated on the fly: any number of rows, none of them stored. Used to show
/// that a fit's working set does not grow with the series.
/// </summary>
public sealed class SyntheticAr1Source(long rows, double phi, int seed, int batchSize = 8192) : ITimeSeriesSource
{
    public int RegressorCount => 0;

    public ValueTask<ITimeSeriesCursor> OpenAsync(CancellationToken cancellationToken = default)
        => new(new Cursor(rows, phi, seed, batchSize));

    private sealed class Cursor(long rows, double phi, int seed, int batchSize) : ITimeSeriesCursor
    {
        private readonly double[] _buffer = new double[batchSize];
        private readonly Random _random = new(seed);
        private long _produced;
        private double _value;

        public SeriesBatch Current { get; private set; }

        public ValueTask<bool> ReadAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var take = (int)Math.Min(batchSize, rows - _produced);
            if (take <= 0)
            {
                return new ValueTask<bool>(false);
            }

            for (var i = 0; i < take; i++)
            {
                // Cheap approximately Gaussian innovation: sum of three uniforms.
                var innovation = _random.NextDouble() + _random.NextDouble() + _random.NextDouble() - 1.5;
                _value = (phi * _value) + innovation;
                _buffer[i] = _value;
            }

            _produced += take;
            Current = new SeriesBatch(_buffer.AsMemory(0, take), ReadOnlyMemory<double>.Empty, 0);
            return new ValueTask<bool>(true);
        }

        public ValueTask DisposeAsync() => default;
    }
}

/// <summary>
/// The contract every <see cref="ITimeSeriesSource"/> must satisfy, checked against known
/// data. Written once, run against every adapter.
/// </summary>
public static class SourceContract
{
    public static async Task VerifyAsync(ITimeSeriesSource source, double[] expected, ExogenousMatrix? exogenous, int maxBatch)
    {
        Assert.Equal(exogenous?.RegressorCount ?? 0, source.RegressorCount);

        for (var pass = 0; pass < 2; pass++)
        {
            var values = new List<double>();
            var regressors = new List<double>();

            var cursor = await source.OpenAsync();
            try
            {
                while (await cursor.ReadAsync())
                {
                    var batch = cursor.Current;
                    Assert.True(batch.Count > 0, "A delivered batch must not be empty.");
                    Assert.True(batch.Count <= maxBatch, $"Batch of {batch.Count} exceeds {maxBatch}.");
                    Assert.Equal(source.RegressorCount, batch.RegressorCount);
                    Assert.Equal(batch.Count * batch.RegressorCount, batch.Exogenous.Length);

                    values.AddRange(batch.Values.ToArray());
                    regressors.AddRange(batch.Exogenous.ToArray());
                }

                // Reading past the end stays at the end.
                Assert.False(await cursor.ReadAsync());
            }
            finally
            {
                await cursor.DisposeAsync();
            }

            Assert.Equal(expected, values);

            if (exogenous is not null)
            {
                Assert.Equal(exogenous.Values.ToArray(), regressors);
            }
        }
    }
}
