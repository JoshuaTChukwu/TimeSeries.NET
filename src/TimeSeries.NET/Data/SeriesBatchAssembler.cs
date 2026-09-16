using System.Buffers;
using System.Globalization;

namespace TimeSeries.Data;

/// <summary>
/// Turns a stream of source rows into <see cref="SeriesBatch"/>es, applying the gap
/// policy, validating time and key order, and cutting batches at key boundaries. Adapter
/// authors feed it one row at a time and hand back whatever it assembles; every adapter
/// that uses it behaves identically at a gap or an out-of-order row.
/// </summary>
/// <remarks>
/// <para>
/// Protocol: call <see cref="BeginBatch"/>, then <see cref="Push"/> rows until it returns
/// false or <see cref="IsFull"/> is true, then hand <see cref="Batch"/> to the consumer
/// and start again. A false return means the row was kept back — the batch is complete
/// because a new key began or a gap is being filled — and it will be emitted by the next
/// <see cref="BeginBatch"/>. When the source is exhausted, keep cycling while
/// <see cref="HasDeferredWork"/> is true; the last deferred row is not lost.
/// </para>
/// <para>
/// Buffers come from <see cref="ArrayPool{T}"/> and are returned on <see cref="Dispose"/>.
/// <see cref="Batch"/> is valid only until the next <see cref="BeginBatch"/>.
/// </para>
/// </remarks>
public sealed class SeriesBatchAssembler : IDisposable
{
    private readonly int _regressors;
    private readonly int _batchSize;
    private readonly bool _grouped;
    private readonly bool _trackTime;
    private readonly bool _timeIsTicks;
    private readonly GapPolicy _policy;
    private readonly double _step;
    private readonly double _maxGap;
    private readonly bool _validateKeyOrder;
    private readonly IComparer<object>? _keyComparer;

    private double[] _values;
    private double[] _exogenous;
    private int _count;
    private bool _disposed;

    private string? _batchKey;
    private object? _currentKey;
    private bool _hasLast;
    private double _lastTime;
    private object? _lastTimeRaw;
    private double _lastValue;
    private readonly double[] _lastExogenous;

    private bool _hasPending;
    private bool _pendingGapHandled;
    private double _pendingValue;
    private readonly double[] _pendingExogenous;
    private double _pendingTime;
    private object? _pendingTimeRaw;
    private object? _pendingKey;

    private int _fillRemaining;
    private int _fillTotal;
    private double _fillTo;
    private readonly double[] _fillToExogenous;

    /// <summary>
    /// Creates an assembler.
    /// </summary>
    /// <param name="regressorCount">Regressors per row.</param>
    /// <param name="batchSize">Rows per batch.</param>
    /// <param name="options">Gap and ordering policy.</param>
    /// <param name="grouped">Whether rows carry a key and batches must not span keys.</param>
    /// <param name="trackTime">Whether rows carry a time for gap detection and order validation.</param>
    /// <param name="timeIsTicks">
    /// Whether times are <see cref="DateTime.Ticks"/> (so spans format as durations) or
    /// plain numbers in the source's own units.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="regressorCount"/> is negative or <paramref name="batchSize"/> is below 1.</exception>
    /// <exception cref="ArgumentException">A fill policy is requested without an expected step.</exception>
    public SeriesBatchAssembler(int regressorCount, int batchSize, DbSourceOptions options, bool grouped, bool trackTime, bool timeIsTicks)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (regressorCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(regressorCount), regressorCount, "Regressor count must be zero or greater.");
        }

        if (batchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be at least 1.");
        }

        _trackTime = trackTime && options.Step is not null;

        if (trackTime && options.Gaps != GapPolicy.Throw && options.Step is null)
        {
            throw new ArgumentException(
                $"GapPolicy.{options.Gaps} needs to know how many observations are missing, so ExpectedStep " +
                "(or ExpectedNumericStep) must be set.",
                nameof(options));
        }

        _regressors = regressorCount;
        _batchSize = batchSize;
        _grouped = grouped;
        _timeIsTicks = timeIsTicks;
        _policy = options.Gaps;
        _step = options.Step ?? 0d;
        _maxGap = options.MaximumGap ?? 0d;
        _validateKeyOrder = options.ValidateKeyOrder;
        _keyComparer = options.KeyComparer;

        _values = ArrayPool<double>.Shared.Rent(batchSize);
        _exogenous = regressorCount == 0 ? [] : ArrayPool<double>.Shared.Rent(batchSize * regressorCount);
        _lastExogenous = new double[regressorCount];
        _pendingExogenous = new double[regressorCount];
        _fillToExogenous = new double[regressorCount];
    }

    /// <summary>Rows in the current batch.</summary>
    public int Count => _count;

    /// <summary>True when the current batch cannot take another row.</summary>
    public bool IsFull => _count >= _batchSize;

    /// <summary>True when a kept-back row or unfinished gap fill is waiting for the next batch.</summary>
    public bool HasDeferredWork => _hasPending || _fillRemaining > 0;

    /// <summary>The current batch. Valid until the next <see cref="BeginBatch"/>.</summary>
    public SeriesBatch Batch => new(_values.AsMemory(0, _count), _exogenous.AsMemory(0, _count * _regressors), _regressors);

    /// <summary>The key every row of the current batch belongs to, or empty when not grouped.</summary>
    public string BatchKey => _batchKey ?? string.Empty;

    /// <summary>
    /// Starts a fresh batch, emitting any deferred gap fills and kept-back row into it first.
    /// </summary>
    public void BeginBatch()
    {
        ThrowIfDisposed();
        _count = 0;

        if (_hasPending)
        {
            _batchKey = KeyText(_pendingKey);
        }

        DrainFills();

        if (_hasPending && _fillRemaining == 0 && !IsFull)
        {
            _hasPending = false;
            Process(_pendingValue, _pendingExogenous, _pendingTime, _pendingTimeRaw, _pendingKey, _pendingGapHandled);
        }
    }

    /// <summary>
    /// Offers one source row.
    /// </summary>
    /// <param name="value">The observation.</param>
    /// <param name="exogenous">Its regressors, exactly the configured count.</param>
    /// <param name="time">Its time as a number (ticks or the source's units); ignored when time is not tracked.</param>
    /// <param name="timeRaw">The time as the source gave it, for messages.</param>
    /// <param name="key">Its key; ignored when not grouped.</param>
    /// <returns>
    /// True when the row went into the current batch and more may follow; false when the
    /// row was kept back and the current batch is complete.
    /// </returns>
    /// <exception cref="InvalidOperationException">The batch is already full; call <see cref="BeginBatch"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="exogenous"/> has the wrong length.</exception>
    /// <exception cref="SeriesGapException">A gap under <see cref="GapPolicy.Throw"/>.</exception>
    /// <exception cref="SeriesOrderException">Time did not advance, or a key arrived out of order.</exception>
    public bool Push(double value, ReadOnlySpan<double> exogenous, double time, object? timeRaw, object? key)
    {
        ThrowIfDisposed();

        if (IsFull || HasDeferredWork)
        {
            throw new InvalidOperationException("The current batch is complete; take it and call BeginBatch before pushing more rows.");
        }

        if (exogenous.Length != _regressors)
        {
            throw new ArgumentException($"Expected {_regressors} regressor values; {exogenous.Length} supplied.", nameof(exogenous));
        }

        Span<double> copy = _regressors <= 32 ? stackalloc double[_regressors] : new double[_regressors];
        exogenous.CopyTo(copy);
        return Process(value, copy, time, timeRaw, key, gapHandled: false);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_values.Length > 0)
        {
            ArrayPool<double>.Shared.Return(_values);
        }

        if (_exogenous.Length > 0)
        {
            ArrayPool<double>.Shared.Return(_exogenous);
        }

        _values = [];
        _exogenous = [];
    }

    private bool Process(double value, ReadOnlySpan<double> exogenous, double time, object? timeRaw, object? key, bool gapHandled)
    {
        if (_grouped)
        {
            if (_batchKey is null)
            {
                _batchKey = KeyText(key);
                _currentKey = key;
            }
            else if (!KeysEqual(key, _currentKey))
            {
                ValidateKeyOrder(_currentKey, key);
                Defer(value, exogenous, time, timeRaw, key, gapHandled: false);
                _currentKey = key;
                _hasLast = false;
                return false;
            }
        }

        if (_trackTime && _hasLast && !gapHandled)
        {
            var delta = time - _lastTime;

            if (delta <= 0d)
            {
                throw new SeriesOrderException(
                    TimeText(_lastTimeRaw), TimeText(timeRaw),
                    $"Timestamps must strictly increase within a series; {TimeText(timeRaw)} follows {TimeText(_lastTimeRaw)}" +
                    $"{(_grouped ? $" for key '{_batchKey}'" : string.Empty)}. Check the ordering and for duplicate rows.");
            }

            if (delta > _maxGap)
            {
                var missing = (int)Math.Round(delta / _step, MidpointRounding.AwayFromZero) - 1;

                if (_policy == GapPolicy.Throw)
                {
                    throw new SeriesGapException(
                        TimeText(_lastTimeRaw), TimeText(timeRaw), _grouped ? _batchKey : null,
                        $"Gap of {DeltaText(delta)} between {TimeText(_lastTimeRaw)} and {TimeText(timeRaw)}" +
                        $"{(_grouped ? $" for key '{_batchKey}'" : string.Empty)}; the expected step is {DeltaText(_step)} " +
                        $"and the largest allowed is {DeltaText(_maxGap)}. Roughly {Math.Max(missing, 1)} observation(s) are missing. " +
                        "Set DbSourceOptions.Gaps to ForwardFill or Interpolate to fill them, or raise MaxGap if this spacing is expected.");
                }

                if (missing > 0)
                {
                    _fillRemaining = missing;
                    _fillTotal = missing;
                    _fillTo = value;
                    exogenous.CopyTo(_fillToExogenous);
                    Defer(value, exogenous, time, timeRaw, key, gapHandled: true);
                    DrainFills();
                    return false;
                }
            }
        }

        Emit(value, exogenous);
        _hasLast = true;
        _lastTime = time;
        _lastTimeRaw = timeRaw;
        _lastValue = value;
        exogenous.CopyTo(_lastExogenous);
        return !IsFull;
    }

    private void Defer(double value, ReadOnlySpan<double> exogenous, double time, object? timeRaw, object? key, bool gapHandled)
    {
        _hasPending = true;
        _pendingGapHandled = gapHandled;
        _pendingValue = value;
        exogenous.CopyTo(_pendingExogenous);
        _pendingTime = time;
        _pendingTimeRaw = timeRaw;
        _pendingKey = key;
    }

    private void DrainFills()
    {
        while (_fillRemaining > 0 && !IsFull)
        {
            var position = _fillTotal - _fillRemaining + 1;
            var fraction = (double)position / (_fillTotal + 1);
            var interpolate = _policy == GapPolicy.Interpolate;
            var index = _count * _regressors;

            _values[_count] = interpolate ? _lastValue + ((_fillTo - _lastValue) * fraction) : _lastValue;

            for (var i = 0; i < _regressors; i++)
            {
                _exogenous[index + i] = interpolate
                    ? _lastExogenous[i] + ((_fillToExogenous[i] - _lastExogenous[i]) * fraction)
                    : _lastExogenous[i];
            }

            _count++;
            _fillRemaining--;
        }
    }

    private void Emit(double value, ReadOnlySpan<double> exogenous)
    {
        _values[_count] = value;
        exogenous.CopyTo(_exogenous.AsSpan(_count * _regressors, _regressors));
        _count++;
    }

    private void ValidateKeyOrder(object? previous, object? current)
    {
        if (!_validateKeyOrder)
        {
            return;
        }

        var comparison = _keyComparer is not null
            ? _keyComparer.Compare(previous!, current!)
            : DefaultCompare(previous, current);

        if (comparison > 0)
        {
            throw new SeriesOrderException(
                KeyText(previous), KeyText(current),
                $"Key '{KeyText(current)}' arrived after key '{KeyText(previous)}', so the rows are not ordered by key. " +
                "A grouped scan needs ordering by key then time — and an index on (key, time) to make that cheap. " +
                "If the store legitimately orders keys differently, supply DbSourceOptions.KeyComparer " +
                "or set ValidateKeyOrder to false.");
        }
    }

    private static int DefaultCompare(object? a, object? b)
    {
        if (a is string sa && b is string sb)
        {
            return StringComparer.Ordinal.Compare(sa, sb);
        }

        if (a is IComparable comparable && b is not null && a.GetType() == b.GetType())
        {
            return comparable.CompareTo(b);
        }

        return StringComparer.Ordinal.Compare(KeyText(a), KeyText(b));
    }

    private static bool KeysEqual(object? a, object? b)
    {
        if (a is string sa && b is string sb)
        {
            return string.Equals(sa, sb, StringComparison.Ordinal);
        }

        return Equals(a, b);
    }

    /// <summary>Renders a key for messages and <see cref="KeyedSeriesBatch.Key"/>, culture-invariantly.</summary>
    /// <param name="key">The key object.</param>
    /// <returns>Its text.</returns>
    public static string KeyText(object? key) => key switch
    {
        null => string.Empty,
        string s => s,
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => key.ToString() ?? string.Empty,
    };

    /// <summary>
    /// Converts a source's time value to a number: ticks for date/time and ISO text, the
    /// value itself for numbers.
    /// </summary>
    /// <param name="value">A <see cref="DateTime"/>, <see cref="DateTimeOffset"/>, ISO-8601 string, or number.</param>
    /// <returns>The time as a double.</returns>
    /// <exception cref="InvalidOperationException">The type is not one of those.</exception>
    public static double ToTime(object value) => value switch
    {
        DateTime dateTime => dateTime.Ticks,
        DateTimeOffset offset => offset.UtcTicks,
        string text => DateTime.Parse(text, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind).Ticks,
        IConvertible convertible => convertible.ToDouble(CultureInfo.InvariantCulture),
        _ => throw new InvalidOperationException($"Cannot read a time from a value of type {value.GetType().Name}."),
    };

    private static string TimeText(object? raw) => raw switch
    {
        null => "?",
        DateTime dateTime => dateTime.ToString("O", CultureInfo.InvariantCulture),
        DateTimeOffset offset => offset.ToString("O", CultureInfo.InvariantCulture),
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => raw.ToString() ?? "?",
    };

    private string DeltaText(double delta) =>
        _timeIsTicks ? TimeSpan.FromTicks((long)delta).ToString() : delta.ToString(CultureInfo.InvariantCulture);

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(SeriesBatchAssembler));
        }
    }
}
