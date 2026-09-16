namespace TimeSeries.Data;

/// <summary>
/// An <see cref="ITimeSeriesSource"/> over in-memory arrays. This is how the synchronous
/// API reaches the same engine as the streaming one without a second code path, and the
/// reference implementation every adapter is held to: any test written against it is a
/// test of the contract the adapters must satisfy.
/// </summary>
public sealed class ArraySource : ITimeSeriesSource
{
    private readonly ReadOnlyMemory<double> _values;
    private readonly ExogenousMatrix? _exogenous;
    private readonly int _batchSize;

    /// <summary>
    /// Wraps a series, optionally with regressors.
    /// </summary>
    /// <param name="values">The series, in time order. Referenced, not copied.</param>
    /// <param name="exogenous">Regressors aligned with the series, or null.</param>
    /// <param name="batchSize">Observations per batch. Default 8192.</param>
    /// <exception cref="ArgumentException"><paramref name="exogenous"/> has a different row count.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="batchSize"/> is below 1.</exception>
    public ArraySource(ReadOnlyMemory<double> values, ExogenousMatrix? exogenous = null, int batchSize = 8192)
    {
        if (batchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be at least 1.");
        }

        if (exogenous is not null && exogenous.Count != values.Length)
        {
            throw new ArgumentException(
                $"The series has {values.Length} observations but the regressors have {exogenous.Count} rows.",
                nameof(exogenous));
        }

        _values = values;
        _exogenous = exogenous;
        _batchSize = batchSize;
    }

    /// <inheritdoc />
    public int RegressorCount => _exogenous?.RegressorCount ?? 0;

    /// <summary>The number of observations.</summary>
    public int Count => _values.Length;

    /// <inheritdoc />
    public ValueTask<ITimeSeriesCursor> OpenAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<ITimeSeriesCursor>(new Cursor(this));
    }

    private sealed class Cursor : ITimeSeriesCursor
    {
        private readonly ArraySource _source;
        private int _position;

        internal Cursor(ArraySource source)
        {
            _source = source;
        }

        public SeriesBatch Current { get; private set; }

        public ValueTask<bool> ReadAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var remaining = _source._values.Length - _position;

            if (remaining <= 0)
            {
                Current = default;
                return new ValueTask<bool>(false);
            }

            var take = Math.Min(_source._batchSize, remaining);
            var values = _source._values.Slice(_position, take);
            var regressors = _source.RegressorCount;
            var exogenous = _source._exogenous is null
                ? ReadOnlyMemory<double>.Empty
                : _source._exogenous.Values.Slice(_position * regressors, take * regressors);

            Current = new SeriesBatch(values, exogenous, regressors);
            _position += take;
            return new ValueTask<bool>(true);
        }

        public ValueTask DisposeAsync() => default;
    }
}
