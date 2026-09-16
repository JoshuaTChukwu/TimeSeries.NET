using System.Buffers;
using System.Data;
using System.Data.Common;
using System.Globalization;

namespace TimeSeries.Data;

/// <summary>
/// One traversal of a <see cref="DbTimeSeriesSource"/>. Reads rows into pooled batch
/// buffers, detects gaps and ordering faults as it goes, and cuts batches at key
/// boundaries so a batch never mixes two series.
/// </summary>
public sealed class DbSeriesCursor : ITimeSeriesCursor, IGroupedTimeSeriesCursor
{
    private readonly DbTimeSeriesSource _source;
    private readonly DbConnection _connection;
    private readonly int _regressors;
    private readonly int _batchSize;
    private readonly bool _grouped;
    private readonly bool _trackTime;
    private readonly double _step;
    private readonly double _maxGap;

    private DbCommand? _command;
    private DbDataReader? _reader;
    private double[] _values = [];
    private double[] _exogenous = [];
    private int _count;
    private bool _disposed;
    private long _rowsRead;

    // Column plumbing, fixed once the reader is open.
    private Column[] _columns = [];
    private double _rowValue;
    private readonly double[] _rowExogenous;
    private double _rowTime;
    private object? _rowTimeRaw;
    private object? _rowKey;

    // Series state, reset at each key change.
    private string? _batchKey;
    private object? _currentKey;
    private bool _hasLast;
    private double _lastTime;
    private object? _lastTimeRaw;
    private double _lastValue;
    private readonly double[] _lastExogenous;

    // A row read from the database but not yet emitted: the first row of the next key,
    // or the row after a gap whose fills are still being emitted.
    private bool _hasPending;
    private bool _pendingGapHandled;
    private double _pendingValue;
    private readonly double[] _pendingExogenous;
    private double _pendingTime;
    private object? _pendingTimeRaw;
    private object? _pendingKey;

    // Fill rows still owed for a gap.
    private int _fillRemaining;
    private int _fillTotal;
    private double _fillTo;
    private readonly double[] _fillToExogenous;

    internal DbSeriesCursor(DbTimeSeriesSource source, DbConnection connection)
    {
        _source = source;
        _connection = connection;
        _regressors = source.RegressorCount;
        _batchSize = source.Query.BatchSize;
        _grouped = source.IsGrouped;
        _trackTime = source.Query.TimeColumn is not null && source.Options.Step is not null;
        _step = source.Options.Step ?? 0d;
        _maxGap = source.Options.MaximumGap ?? 0d;

        _rowExogenous = new double[_regressors];
        _lastExogenous = new double[_regressors];
        _pendingExogenous = new double[_regressors];
        _fillToExogenous = new double[_regressors];
    }

    private enum Role
    {
        Value,
        Exogenous,
        Time,
        Key,
    }

    private readonly struct Column(int ordinal, Role role, int index, Func<DbDataReader, int, double>? read)
    {
        public int Ordinal => ordinal;

        public Role Role => role;

        public int Index => index;

        public Func<DbDataReader, int, double>? Read => read;
    }

    /// <summary>Rows read from the database so far, before any fills.</summary>
    public long RowsRead => _rowsRead;

    /// <summary>The current batch. Valid until the next <see cref="ReadAsync"/>.</summary>
    public SeriesBatch Current =>
        new(_values.AsMemory(0, _count), _exogenous.AsMemory(0, _count * _regressors), _regressors);

    /// <inheritdoc />
    KeyedSeriesBatch IGroupedTimeSeriesCursor.Current => new(_batchKey ?? string.Empty, Current);

    /// <inheritdoc />
    SeriesBatch ITimeSeriesCursor.Current => Current;

    internal async ValueTask OpenAsync(CancellationToken cancellationToken)
    {
        if (_connection.State != ConnectionState.Open)
        {
            await _connection.OpenAsync(cancellationToken).ConfigureAwait(false);
        }

        var query = _source.Query;
        _command = _connection.CreateCommand();
        _command.CommandText = query.CommandText;

        if (query.CommandTimeoutSeconds is int timeout)
        {
            _command.CommandTimeout = timeout;
        }

        foreach (var pair in query.Parameters)
        {
            var parameter = _command.CreateParameter();
            parameter.ParameterName = pair.Key;
            parameter.Value = pair.Value ?? DBNull.Value;
            _command.Parameters.Add(parameter);
        }

        _reader = await _command.ExecuteReaderAsync(CommandBehavior.SequentialAccess, cancellationToken).ConfigureAwait(false);

        var columns = new List<Column>
        {
            new(_reader.GetOrdinal(query.ValueColumn), Role.Value, 0, NumericReader(_reader, _reader.GetOrdinal(query.ValueColumn))),
        };

        for (var i = 0; i < query.ExogenousColumns.Count; i++)
        {
            var ordinal = _reader.GetOrdinal(query.ExogenousColumns[i]);
            columns.Add(new Column(ordinal, Role.Exogenous, i, NumericReader(_reader, ordinal)));
        }

        if (query.TimeColumn is not null)
        {
            columns.Add(new Column(_reader.GetOrdinal(query.TimeColumn), Role.Time, 0, null));
        }

        if (query.KeyColumn is not null)
        {
            columns.Add(new Column(_reader.GetOrdinal(query.KeyColumn), Role.Key, 0, null));
        }

        // Sequential access requires ascending ordinal order.
        _columns = columns.OrderBy(c => c.Ordinal).ToArray();

        _values = ArrayPool<double>.Shared.Rent(_batchSize);
        _exogenous = _regressors == 0 ? [] : ArrayPool<double>.Shared.Rent(_batchSize * _regressors);
    }

    /// <inheritdoc />
    public async ValueTask<bool> ReadAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(DbSeriesCursor));
        }

        cancellationToken.ThrowIfCancellationRequested();
        _count = 0;

        if (_hasPending)
        {
            _batchKey = KeyText(_pendingKey);
        }

        DrainFills();

        if (_hasPending && _fillRemaining == 0 && _count < _batchSize)
        {
            _hasPending = false;
            Process(_pendingValue, _pendingExogenous, _pendingTime, _pendingTimeRaw, _pendingKey, _pendingGapHandled);
        }

        while (_count < _batchSize && !_hasPending && _fillRemaining == 0
            && await _reader!.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            ReadRow();
            _rowsRead++;
            Process(_rowValue, _rowExogenous, _rowTime, _rowTimeRaw, _rowKey, gapHandled: false);
        }

        return _count > 0;
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
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

#if NET8_0_OR_GREATER
        if (_reader is not null)
        {
            await _reader.DisposeAsync().ConfigureAwait(false);
        }

        if (_command is not null)
        {
            await _command.DisposeAsync().ConfigureAwait(false);
        }

        await _connection.DisposeAsync().ConfigureAwait(false);
#else
        _reader?.Dispose();
        _command?.Dispose();
        _connection.Dispose();
        await Task.CompletedTask.ConfigureAwait(false);
#endif
    }

    private void ReadRow()
    {
        var reader = _reader!;

        foreach (var column in _columns)
        {
            switch (column.Role)
            {
                case Role.Value:
                    _rowValue = ReadNumeric(reader, column, _source.Query.ValueColumn);
                    break;

                case Role.Exogenous:
                    _rowExogenous[column.Index] = ReadNumeric(reader, column, _source.Query.ExogenousColumns[column.Index]);
                    break;

                case Role.Time:
                    _rowTimeRaw = reader.IsDBNull(column.Ordinal) ? null : reader.GetValue(column.Ordinal);
                    _rowTime = _rowTimeRaw is null
                        ? throw new InvalidSeriesException(_source.Query.TimeColumn!, _rowsRead, double.NaN)
                        : ToTime(_rowTimeRaw);
                    break;

                case Role.Key:
                    _rowKey = reader.IsDBNull(column.Ordinal) ? null : reader.GetValue(column.Ordinal);
                    break;
            }
        }
    }

    private double ReadNumeric(DbDataReader reader, Column column, string name)
    {
        if (reader.IsDBNull(column.Ordinal))
        {
            throw new InvalidSeriesException(name, _rowsRead, double.NaN);
        }

        var value = column.Read!(reader, column.Ordinal);

        if (double.IsNaN(value) || double.IsInfinity(value))
        {
            throw new InvalidSeriesException(name, _rowsRead, value);
        }

        return value;
    }

    private void Process(double value, double[] exogenous, double time, object? timeRaw, object? key, bool gapHandled)
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
                return;
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
                    $"{(_grouped ? $" for key '{_batchKey}'" : string.Empty)}. Check the ORDER BY and for duplicate rows.");
            }

            if (delta > _maxGap)
            {
                var missing = (int)Math.Round(delta / _step, MidpointRounding.AwayFromZero) - 1;

                if (_source.Options.Gaps == GapPolicy.Throw)
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
                    Array.Copy(exogenous, _fillToExogenous, _regressors);
                    Defer(value, exogenous, time, timeRaw, key, gapHandled: true);
                    DrainFills();
                    return;
                }
            }
        }

        Emit(value, exogenous);
        _hasLast = true;
        _lastTime = time;
        _lastTimeRaw = timeRaw;
        _lastValue = value;
        Array.Copy(exogenous, _lastExogenous, _regressors);
    }

    private void Defer(double value, double[] exogenous, double time, object? timeRaw, object? key, bool gapHandled)
    {
        _hasPending = true;
        _pendingGapHandled = gapHandled;
        _pendingValue = value;
        Array.Copy(exogenous, _pendingExogenous, _regressors);
        _pendingTime = time;
        _pendingTimeRaw = timeRaw;
        _pendingKey = key;
    }

    private void DrainFills()
    {
        while (_fillRemaining > 0 && _count < _batchSize)
        {
            var position = _fillTotal - _fillRemaining + 1;
            var fraction = (double)position / (_fillTotal + 1);
            var interpolate = _source.Options.Gaps == GapPolicy.Interpolate;

            var value = interpolate ? _lastValue + ((_fillTo - _lastValue) * fraction) : _lastValue;
            var index = _count * _regressors;

            _values[_count] = value;
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

    private void Emit(double value, double[] exogenous)
    {
        _values[_count] = value;
        Array.Copy(exogenous, 0, _exogenous, _count * _regressors, _regressors);
        _count++;
    }

    private void ValidateKeyOrder(object? previous, object? current)
    {
        if (!_source.Options.ValidateKeyOrder)
        {
            return;
        }

        var comparison = _source.Options.KeyComparer is IComparer<object> comparer
            ? comparer.Compare(previous!, current!)
            : DefaultCompare(previous, current);

        if (comparison > 0)
        {
            throw new SeriesOrderException(
                KeyText(previous), KeyText(current),
                $"Key '{KeyText(current)}' arrived after key '{KeyText(previous)}', so the result set is not ordered by key. " +
                "A grouped scan needs ORDER BY key, time — and a composite index on (key, time) to make that cheap. " +
                "If the database collation legitimately orders keys differently, supply DbSourceOptions.KeyComparer " +
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

    private static string KeyText(object? key) => key switch
    {
        null => string.Empty,
        string s => s,
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => key.ToString() ?? string.Empty,
    };

    private static double ToTime(object value) => value switch
    {
        DateTime dateTime => dateTime.Ticks,
        DateTimeOffset offset => offset.UtcTicks,
        string text => DateTime.Parse(text, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind).Ticks,
        IConvertible convertible => convertible.ToDouble(CultureInfo.InvariantCulture),
        _ => throw new InvalidOperationException($"Cannot read a time from a column of type {value.GetType().Name}."),
    };

    private string TimeText(object? raw) => raw switch
    {
        null => "?",
        DateTime dateTime => dateTime.ToString("O", CultureInfo.InvariantCulture),
        DateTimeOffset offset => offset.ToString("O", CultureInfo.InvariantCulture),
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => raw.ToString() ?? "?",
    };

    private string DeltaText(double delta) =>
        _source.Options.ExpectedStep is not null || _source.Options.MaxGap is not null
            ? TimeSpan.FromTicks((long)delta).ToString()
            : delta.ToString(CultureInfo.InvariantCulture);

    /// <summary>
    /// Picks a typed reader for a numeric column once, from its declared type, rather than
    /// boxing every cell through <see cref="DbDataReader.GetValue"/>.
    /// </summary>
    private static Func<DbDataReader, int, double> NumericReader(DbDataReader reader, int ordinal)
    {
        var type = reader.GetFieldType(ordinal);

        if (type == typeof(double))
        {
            return static (r, o) => r.GetDouble(o);
        }

        if (type == typeof(float))
        {
            return static (r, o) => r.GetFloat(o);
        }

        if (type == typeof(decimal))
        {
            return static (r, o) => (double)r.GetDecimal(o);
        }

        if (type == typeof(long))
        {
            return static (r, o) => r.GetInt64(o);
        }

        if (type == typeof(int))
        {
            return static (r, o) => r.GetInt32(o);
        }

        if (type == typeof(short))
        {
            return static (r, o) => r.GetInt16(o);
        }

        if (type == typeof(byte))
        {
            return static (r, o) => r.GetByte(o);
        }

        return static (r, o) => Convert.ToDouble(r.GetValue(o), CultureInfo.InvariantCulture);
    }
}
