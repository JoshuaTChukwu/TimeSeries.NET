using MongoDB.Bson;
using MongoDB.Driver;

namespace TimeSeries.Data.MongoDb;

/// <summary>
/// Which documents to read and which fields mean what. Either a filter with a sort, or an
/// aggregation pipeline whose output documents carry the named fields.
/// </summary>
/// <remarks>
/// Documents must arrive ordered by time — and by key, then time, when
/// <see cref="KeyField"/> is set. With a filter, the default sort does that; with a
/// pipeline, the pipeline must end in a matching <c>$sort</c>, and the collection wants an
/// index on <c>(key, time)</c> so the server does not sort on disk.
/// </remarks>
public sealed record MongoSeriesQuery
{
    /// <summary>
    /// The field holding the observed value. Default <c>value</c>. Null makes this a
    /// regressor-only query for <see cref="ExogenousMatrix.FromSourceAsync"/>.
    /// </summary>
    public string? ValueField { get; init; } = "value";

    /// <summary>Fields holding exogenous regressors, in the order the model will see them.</summary>
    public IReadOnlyList<string> ExogenousFields { get; init; } = [];

    /// <summary>
    /// The field holding each observation's time — a BSON date, ISO-8601 string, or number.
    /// Null disables gap detection and order validation.
    /// </summary>
    public string? TimeField { get; init; }

    /// <summary>The field identifying which series a document belongs to. Setting it makes the source a population.</summary>
    public string? KeyField { get; init; }

    /// <summary>Documents to include. Default all. Ignored when <see cref="Pipeline"/> is set.</summary>
    public FilterDefinition<BsonDocument>? Filter { get; init; }

    /// <summary>
    /// The sort. Default: ascending <see cref="KeyField"/> then <see cref="TimeField"/>, or
    /// natural order when no time field is set. Ignored when <see cref="Pipeline"/> is set.
    /// </summary>
    public SortDefinition<BsonDocument>? Sort { get; init; }

    /// <summary>An aggregation pipeline to read from instead of a find. Its output must be ordered.</summary>
    public PipelineDefinition<BsonDocument, BsonDocument>? Pipeline { get; init; }

    /// <summary>Documents per driver batch and rows per estimator batch. Default 8192.</summary>
    public int BatchSize { get; init; } = 8192;

    /// <summary>Checks the query.</summary>
    /// <exception cref="ArgumentException">A field name is blank or duplicated.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="BatchSize"/> is below 1.</exception>
    public void Validate()
    {
        if (ValueField is not null && string.IsNullOrWhiteSpace(ValueField))
        {
            throw new ArgumentException("ValueField must name a field, or be null for a regressor-only query.", nameof(ValueField));
        }

        if (ValueField is null && ExogenousFields.Count == 0)
        {
            throw new ArgumentException("A query needs a ValueField, ExogenousFields, or both.", nameof(ValueField));
        }

        if (BatchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(BatchSize), BatchSize, "Batch size must be at least 1.");
        }

        var names = new HashSet<string>(StringComparer.Ordinal);

        if (ValueField is not null)
        {
            names.Add(ValueField);
        }

        foreach (var field in ExogenousFields)
        {
            if (string.IsNullOrWhiteSpace(field) || !names.Add(field))
            {
                throw new ArgumentException($"Exogenous field '{field}' is blank or duplicates another field.", nameof(ExogenousFields));
            }
        }

        if (TimeField is not null && (string.IsNullOrWhiteSpace(TimeField) || !names.Add(TimeField)))
        {
            throw new ArgumentException($"TimeField '{TimeField}' is blank or duplicates another field.", nameof(TimeField));
        }

        if (KeyField is not null && (string.IsNullOrWhiteSpace(KeyField) || !names.Add(KeyField)))
        {
            throw new ArgumentException($"KeyField '{KeyField}' is blank or duplicates another field.", nameof(KeyField));
        }
    }

    internal SortDefinition<BsonDocument>? EffectiveSort()
    {
        if (Sort is not null)
        {
            return Sort;
        }

        if (TimeField is null && KeyField is null)
        {
            return null;
        }

        var builder = Builders<BsonDocument>.Sort;
        SortDefinition<BsonDocument>? sort = null;

        if (KeyField is not null)
        {
            sort = builder.Ascending(KeyField);
        }

        if (TimeField is not null)
        {
            sort = sort is null ? builder.Ascending(TimeField) : sort.Ascending(TimeField);
        }

        return sort;
    }
}
