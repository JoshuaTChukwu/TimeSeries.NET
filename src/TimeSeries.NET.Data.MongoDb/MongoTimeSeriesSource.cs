using MongoDB.Bson;
using MongoDB.Driver;

namespace TimeSeries.Data.MongoDb;

/// <summary>
/// Streams a series — or a keyed population — out of a MongoDB collection through the
/// driver's batched <see cref="IAsyncCursor{TDocument}"/>, without materialising it.
/// </summary>
/// <remarks>
/// The driver already delivers documents in batches, so each server round trip maps onto
/// one estimator batch. Gap policy, time and key ordering are applied by the same
/// <see cref="SeriesBatchAssembler"/> the SQL adapter uses, so the two behave identically.
/// </remarks>
public sealed class MongoTimeSeriesSource : ITimeSeriesSource, IGroupedTimeSeriesSource
{
    private readonly Func<CancellationToken, Task<IAsyncCursor<BsonDocument>>> _open;

    /// <summary>
    /// Creates a source over a collection.
    /// </summary>
    /// <param name="collection">The collection to read.</param>
    /// <param name="query">Which documents and fields.</param>
    /// <param name="options">Spacing and ordering policy; null for the defaults.</param>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    /// <exception cref="ArgumentException">The query or gap configuration is inconsistent.</exception>
    public MongoTimeSeriesSource(IMongoCollection<BsonDocument> collection, MongoSeriesQuery query, DbSourceOptions? options = null)
        : this(query, options, cancellationToken => OpenCursorAsync(collection ?? throw new ArgumentNullException(nameof(collection)), query, cancellationToken))
    {
    }

    /// <summary>For tests: a source whose cursor comes from anywhere.</summary>
    internal MongoTimeSeriesSource(MongoSeriesQuery query, DbSourceOptions? options, Func<CancellationToken, Task<IAsyncCursor<BsonDocument>>> open)
    {
        Query = query ?? throw new ArgumentNullException(nameof(query));
        Options = options ?? DbSourceOptions.Default;
        _open = open;

        Query.Validate();
        using var probe = new SeriesBatchAssembler(RegressorCount, Query.BatchSize, Options, IsGrouped, Query.TimeField is not null, timeIsTicks: true);
    }

    /// <summary>Which documents and fields.</summary>
    public MongoSeriesQuery Query { get; }

    /// <summary>Spacing and ordering policy.</summary>
    public DbSourceOptions Options { get; }

    /// <inheritdoc cref="ITimeSeriesSource.RegressorCount" />
    public int RegressorCount => Query.ExogenousFields.Count;

    /// <summary>True when a <see cref="MongoSeriesQuery.KeyField"/> is set.</summary>
    public bool IsGrouped => Query.KeyField is not null;

    /// <inheritdoc />
    async ValueTask<ITimeSeriesCursor> ITimeSeriesSource.OpenAsync(CancellationToken cancellationToken)
    {
        if (IsGrouped)
        {
            throw new InvalidOperationException(
                $"The query has KeyField '{Query.KeyField}', so it describes a population of series. " +
                "Fit it with ArimaModel.FitManyAsync, or remove the key field to fit a single series.");
        }

        return await OpenAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    async ValueTask<IGroupedTimeSeriesCursor> IGroupedTimeSeriesSource.OpenAsync(CancellationToken cancellationToken)
    {
        if (!IsGrouped)
        {
            throw new InvalidOperationException("The query has no KeyField, so it describes a single series. Set KeyField to scan a population.");
        }

        return await OpenAsync(cancellationToken).ConfigureAwait(false);
    }

    private async ValueTask<MongoSeriesCursor> OpenAsync(CancellationToken cancellationToken)
    {
        var cursor = await _open(cancellationToken).ConfigureAwait(false);
        return new MongoSeriesCursor(this, cursor);
    }

    private static async Task<IAsyncCursor<BsonDocument>> OpenCursorAsync(
        IMongoCollection<BsonDocument> collection, MongoSeriesQuery query, CancellationToken cancellationToken)
    {
        if (query.Pipeline is not null)
        {
            return await collection.AggregateAsync(
                query.Pipeline,
                new AggregateOptions { BatchSize = query.BatchSize, AllowDiskUse = true },
                cancellationToken).ConfigureAwait(false);
        }

        var options = new FindOptions<BsonDocument> { BatchSize = query.BatchSize, Sort = query.EffectiveSort() };
        return await collection.FindAsync(query.Filter ?? FilterDefinition<BsonDocument>.Empty, options, cancellationToken).ConfigureAwait(false);
    }
}
