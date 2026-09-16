namespace TimeSeries.Data;

/// <summary>
/// The query a <see cref="DbTimeSeriesSource"/> runs and which columns of its result set
/// mean what. The library supplies the cursor; the consumer supplies the SQL — anything
/// more becomes an ORM that is wrong for somebody's schema.
/// </summary>
/// <remarks>
/// <para>
/// The result set must be ordered by time — and by key, then time, when
/// <see cref="KeyColumn"/> is set. Prefer holding one ordered cursor open over paging;
/// if paging is unavoidable, use keyset seeks, never <c>OFFSET</c>, which is
/// <c>O(n)</c> per page and turns a scan into <c>O(n^2)</c>.
/// </para>
/// <para>
/// Parameter names are passed to the provider verbatim. Most providers accept a bare
/// name and prefix it themselves; Microsoft.Data.Sqlite requires the prefix in the name
/// (<c>"@from"</c>), so include it when the provider needs it.
/// </para>
/// </remarks>
public sealed record DbSeriesQuery
{
    /// <summary>The SQL to execute. Required.</summary>
    public string CommandText { get; init; } = string.Empty;

    /// <summary>The column holding the observed value. Default <c>value</c>.</summary>
    public string ValueColumn { get; init; } = "value";

    /// <summary>Columns holding exogenous regressors, in the order the model will see them.</summary>
    public IReadOnlyList<string> ExogenousColumns { get; init; } = [];

    /// <summary>
    /// The column holding each observation's time, used for gap detection and order
    /// validation. Null disables both. Accepts date/time, ISO-8601 text, or numeric columns.
    /// </summary>
    public string? TimeColumn { get; init; }

    /// <summary>
    /// The column identifying which series a row belongs to. Setting it makes the source
    /// a population for <see cref="ArimaModel.FitManyAsync"/>; the result set must then be
    /// ordered by this column first.
    /// </summary>
    public string? KeyColumn { get; init; }

    /// <summary>Command parameters by name. Values are passed to the provider as given.</summary>
    public IDictionary<string, object?> Parameters { get; } = new Dictionary<string, object?>(StringComparer.Ordinal);

    /// <summary>Rows per batch handed to the estimator. Default 8192.</summary>
    public int BatchSize { get; init; } = 8192;

    /// <summary>Command timeout in seconds, or null for the provider's default.</summary>
    public int? CommandTimeoutSeconds { get; init; }

    /// <summary>Checks the query for consistency.</summary>
    /// <exception cref="ArgumentException">Required text is missing or a column name is blank or duplicated.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="BatchSize"/> is below 1.</exception>
    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(CommandText))
        {
            throw new ArgumentException("CommandText is required.", nameof(CommandText));
        }

        if (string.IsNullOrWhiteSpace(ValueColumn))
        {
            throw new ArgumentException("ValueColumn is required.", nameof(ValueColumn));
        }

        if (BatchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(BatchSize), BatchSize, "Batch size must be at least 1.");
        }

        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ValueColumn };

        foreach (var column in ExogenousColumns)
        {
            if (string.IsNullOrWhiteSpace(column) || !names.Add(column))
            {
                throw new ArgumentException($"Exogenous column '{column}' is blank or duplicates another column.", nameof(ExogenousColumns));
            }
        }

        if (TimeColumn is not null && (string.IsNullOrWhiteSpace(TimeColumn) || !names.Add(TimeColumn)))
        {
            throw new ArgumentException($"TimeColumn '{TimeColumn}' is blank or duplicates another column.", nameof(TimeColumn));
        }

        if (KeyColumn is not null && (string.IsNullOrWhiteSpace(KeyColumn) || !names.Add(KeyColumn)))
        {
            throw new ArgumentException($"KeyColumn '{KeyColumn}' is blank or duplicates another column.", nameof(KeyColumn));
        }
    }
}
