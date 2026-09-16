using System.Globalization;
using MongoDB.Bson;
using MongoDB.Driver;

namespace TimeSeries.Data.MongoDb;

/// <summary>One traversal of a <see cref="MongoTimeSeriesSource"/>.</summary>
public sealed class MongoSeriesCursor : ITimeSeriesCursor, IGroupedTimeSeriesCursor
{
    private readonly MongoTimeSeriesSource _source;
    private readonly IAsyncCursor<BsonDocument> _cursor;
    private readonly SeriesBatchAssembler _assembler;
    private readonly double[] _rowExogenous;
    private IEnumerator<BsonDocument>? _documents;
    private bool _exhausted;
    private bool _disposed;
    private long _documentsRead;

    internal MongoSeriesCursor(MongoTimeSeriesSource source, IAsyncCursor<BsonDocument> cursor)
    {
        _source = source;
        _cursor = cursor;
        _rowExogenous = new double[source.RegressorCount];

        var timeIsTicks = source.Options.ExpectedStep is not null || source.Options.MaxGap is not null;
        _assembler = new SeriesBatchAssembler(
            source.RegressorCount, source.Query.BatchSize, source.Options, source.IsGrouped,
            source.Query.TimeField is not null, timeIsTicks);
    }

    /// <summary>Documents read so far, before any gap fills.</summary>
    public long DocumentsRead => _documentsRead;

    /// <summary>The current batch. Valid until the next <see cref="ReadAsync"/>.</summary>
    public SeriesBatch Current => _assembler.Batch;

    /// <inheritdoc />
    KeyedSeriesBatch IGroupedTimeSeriesCursor.Current => new(_assembler.BatchKey, Current);

    /// <inheritdoc />
    SeriesBatch ITimeSeriesCursor.Current => Current;

    /// <inheritdoc />
    public async ValueTask<bool> ReadAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MongoSeriesCursor));
        }

        cancellationToken.ThrowIfCancellationRequested();

        while (true)
        {
            _assembler.BeginBatch();

            while (!_assembler.IsFull && !_assembler.HasDeferredWork)
            {
                var document = await NextDocumentAsync(cancellationToken).ConfigureAwait(false);

                if (document is null)
                {
                    break;
                }

                _documentsRead++;
                var (value, time, timeRaw, key) = ReadDocument(document);

                if (!_assembler.Push(value, _rowExogenous, time, timeRaw, key))
                {
                    break;
                }
            }

            if (_assembler.Count > 0)
            {
                return true;
            }

            if (!_assembler.HasDeferredWork)
            {
                return false;
            }
        }
    }

    /// <inheritdoc />
    public ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return default;
        }

        _disposed = true;
        _documents?.Dispose();
        _assembler.Dispose();
        _cursor.Dispose();
        return default;
    }

    /// <summary>The next document across driver batches, or null at the end.</summary>
    private async ValueTask<BsonDocument?> NextDocumentAsync(CancellationToken cancellationToken)
    {
        while (true)
        {
            if (_documents is not null && _documents.MoveNext())
            {
                return _documents.Current;
            }

            if (_exhausted)
            {
                return null;
            }

            _documents?.Dispose();
            _documents = null;

            if (!await _cursor.MoveNextAsync(cancellationToken).ConfigureAwait(false))
            {
                _exhausted = true;
                return null;
            }

            _documents = _cursor.Current.GetEnumerator();
        }
    }

    private (double Value, double Time, object? TimeRaw, object? Key) ReadDocument(BsonDocument document)
    {
        var query = _source.Query;
        var value = ReadNumber(document, query.ValueField);

        for (var i = 0; i < _rowExogenous.Length; i++)
        {
            _rowExogenous[i] = ReadNumber(document, query.ExogenousFields[i]);
        }

        var time = 0d;
        object? timeRaw = null;

        if (query.TimeField is not null)
        {
            if (!document.TryGetValue(query.TimeField, out var bson) || bson.IsBsonNull)
            {
                throw new InvalidSeriesException(query.TimeField, _documentsRead - 1, double.NaN);
            }

            timeRaw = bson.BsonType switch
            {
                BsonType.DateTime => bson.ToUniversalTime(),
                BsonType.String => bson.AsString,
                _ => bson.IsNumeric ? bson.ToDouble() : throw new InvalidOperationException($"Field '{query.TimeField}' is a {bson.BsonType}, not a time."),
            };
            time = SeriesBatchAssembler.ToTime(timeRaw);
        }

        object? key = null;

        if (query.KeyField is not null)
        {
            key = document.TryGetValue(query.KeyField, out var bson) && !bson.IsBsonNull ? KeyOf(bson) : null;
        }

        return (value, time, timeRaw, key);
    }

    private double ReadNumber(BsonDocument document, string field)
    {
        if (!document.TryGetValue(field, out var bson) || bson.IsBsonNull)
        {
            throw new InvalidSeriesException(field, _documentsRead - 1, double.NaN);
        }

        var value = bson.BsonType switch
        {
            BsonType.Double => bson.AsDouble,
            BsonType.Int32 => bson.AsInt32,
            BsonType.Int64 => bson.AsInt64,
            BsonType.Decimal128 => (double)bson.AsDecimal,
            BsonType.String => double.Parse(bson.AsString, NumberStyles.Float, CultureInfo.InvariantCulture),
            _ => throw new InvalidOperationException($"Field '{field}' is a {bson.BsonType}, not a number."),
        };

        if (double.IsNaN(value) || double.IsInfinity(value))
        {
            throw new InvalidSeriesException(field, _documentsRead - 1, value);
        }

        return value;
    }

    private static object KeyOf(BsonValue bson) => bson.BsonType switch
    {
        BsonType.String => bson.AsString,
        BsonType.Int32 => bson.AsInt32,
        BsonType.Int64 => bson.AsInt64,
        BsonType.ObjectId => bson.AsObjectId.ToString(),
        _ => bson.ToString() ?? string.Empty,
    };
}
