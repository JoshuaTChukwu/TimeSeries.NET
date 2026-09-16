using System.Data;
using System.Data.Common;
using System.Globalization;

namespace TimeSeries.Data;

/// <summary>
/// One traversal of a <see cref="DbTimeSeriesSource"/>: reads rows with sequential
/// access through typed column readers and hands them to a
/// <see cref="SeriesBatchAssembler"/>, which applies the gap policy and cuts batches at
/// key boundaries.
/// </summary>
public sealed class DbSeriesCursor : ITimeSeriesCursor, IGroupedTimeSeriesCursor
{
    private readonly DbTimeSeriesSource _source;
    private readonly DbConnection _connection;
    private readonly int _regressors;
    private readonly bool _trackTime;
    private readonly double[] _rowExogenous;

    private DbCommand? _command;
    private DbDataReader? _reader;
    private SeriesBatchAssembler? _assembler;
    private Column[] _columns = [];
    private bool _disposed;
    private long _rowsRead;

    private double _rowValue;
    private double _rowTime;
    private object? _rowTimeRaw;
    private object? _rowKey;

    internal DbSeriesCursor(DbTimeSeriesSource source, DbConnection connection)
    {
        _source = source;
        _connection = connection;
        _regressors = source.RegressorCount;
        _trackTime = source.Query.TimeColumn is not null;
        _rowExogenous = new double[_regressors];
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

    /// <summary>Rows read from the database so far, before any gap fills.</summary>
    public long RowsRead => _rowsRead;

    /// <summary>The current batch. Valid until the next <see cref="ReadAsync"/>.</summary>
    public SeriesBatch Current => _assembler?.Batch ?? default;

    /// <inheritdoc />
    KeyedSeriesBatch IGroupedTimeSeriesCursor.Current => new(_assembler?.BatchKey ?? string.Empty, Current);

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

        var columns = new List<Column>();

        if (query.ValueColumn is not null)
        {
            var valueOrdinal = _reader.GetOrdinal(query.ValueColumn);
            columns.Add(new Column(valueOrdinal, Role.Value, 0, NumericReader(_reader, valueOrdinal)));
        }

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

        var timeIsTicks = _source.Options.ExpectedStep is not null || _source.Options.MaxGap is not null;
        _assembler = new SeriesBatchAssembler(
            _regressors, query.BatchSize, _source.Options, _source.IsGrouped, _trackTime, timeIsTicks);
    }

    /// <inheritdoc />
    public async ValueTask<bool> ReadAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(DbSeriesCursor));
        }

        cancellationToken.ThrowIfCancellationRequested();
        var assembler = _assembler!;

        while (true)
        {
            assembler.BeginBatch();

            while (!assembler.IsFull && !assembler.HasDeferredWork
                && await _reader!.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                ReadRow();
                _rowsRead++;

                if (!assembler.Push(_rowValue, _rowExogenous, _rowTime, _rowTimeRaw, _rowKey))
                {
                    break;
                }
            }

            if (assembler.Count > 0)
            {
                return true;
            }

            if (!assembler.HasDeferredWork)
            {
                return false;
            }
        }
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _assembler?.Dispose();

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
                    _rowValue = ReadNumeric(reader, column, _source.Query.ValueColumn!);
                    break;

                case Role.Exogenous:
                    _rowExogenous[column.Index] = ReadNumeric(reader, column, _source.Query.ExogenousColumns[column.Index]);
                    break;

                case Role.Time:
                    _rowTimeRaw = reader.IsDBNull(column.Ordinal) ? null : reader.GetValue(column.Ordinal);
                    _rowTime = _rowTimeRaw is null
                        ? throw new InvalidSeriesException(_source.Query.TimeColumn!, _rowsRead, double.NaN)
                        : SeriesBatchAssembler.ToTime(_rowTimeRaw);
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
