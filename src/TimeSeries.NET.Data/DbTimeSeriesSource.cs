using System.Data.Common;

namespace TimeSeries.Data;

/// <summary>
/// Streams a series — or a keyed population of series — out of any ADO.NET provider
/// through <see cref="DbDataReader"/>, in batches, without materialising it.
/// </summary>
/// <remarks>
/// <para>
/// Npgsql, Microsoft.Data.SqlClient, Microsoft.Data.Sqlite, MySqlConnector,
/// Oracle.ManagedDataAccess, ClickHouse.Client and DuckDB.NET all derive from
/// <see cref="DbDataReader"/>, so this one adapter reaches every one of them and
/// references none. The consumer's own driver is the only dependency, and they already have it.
/// </para>
/// <para>
/// Each traversal opens a connection from the factory, runs the query with
/// <see cref="System.Data.CommandBehavior.SequentialAccess"/>, reads columns in ordinal
/// order into pooled buffers, and disposes the reader, command and connection when the
/// cursor is disposed — which the estimator does before it starts solving.
/// </para>
/// </remarks>
public sealed class DbTimeSeriesSource : ITimeSeriesSource, IGroupedTimeSeriesSource
{
    private readonly Func<DbConnection> _connectionFactory;

    /// <summary>
    /// Creates a source.
    /// </summary>
    /// <param name="connectionFactory">
    /// Returns a new, unopened connection each time it is called. The source opens it,
    /// owns it for the traversal, and disposes it.
    /// </param>
    /// <param name="query">The query and the roles of its columns.</param>
    /// <param name="options">Spacing and ordering policy; null for the defaults.</param>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    /// <exception cref="ArgumentException">
    /// The query is inconsistent, or a fill policy is requested without an expected step.
    /// </exception>
    public DbTimeSeriesSource(Func<DbConnection> connectionFactory, DbSeriesQuery query, DbSourceOptions? options = null)
    {
        _connectionFactory = connectionFactory ?? throw new ArgumentNullException(nameof(connectionFactory));
        Query = query ?? throw new ArgumentNullException(nameof(query));
        Options = options ?? DbSourceOptions.Default;

        Query.Validate();

        if (Query.TimeColumn is not null && Options.Gaps != GapPolicy.Throw && Options.Step is null)
        {
            throw new ArgumentException(
                $"GapPolicy.{Options.Gaps} needs to know how many observations are missing, so ExpectedStep " +
                "(or ExpectedNumericStep) must be set.",
                nameof(options));
        }
    }

    /// <summary>The query and column roles.</summary>
    public DbSeriesQuery Query { get; }

    /// <summary>Spacing and ordering policy.</summary>
    public DbSourceOptions Options { get; }

    /// <inheritdoc cref="ITimeSeriesSource.RegressorCount" />
    public int RegressorCount => Query.ExogenousColumns.Count;

    /// <summary>True when a <see cref="DbSeriesQuery.KeyColumn"/> is set and the source describes a population.</summary>
    public bool IsGrouped => Query.KeyColumn is not null;

    /// <inheritdoc />
    async ValueTask<ITimeSeriesCursor> ITimeSeriesSource.OpenAsync(CancellationToken cancellationToken)
    {
        if (IsGrouped)
        {
            throw new InvalidOperationException(
                $"The query has KeyColumn '{Query.KeyColumn}', so it describes a population of series. " +
                "Fit it with ArimaModel.FitManyAsync, or remove the key column to fit a single series.");
        }

        return await OpenCursorAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    async ValueTask<IGroupedTimeSeriesCursor> IGroupedTimeSeriesSource.OpenAsync(CancellationToken cancellationToken)
    {
        if (!IsGrouped)
        {
            throw new InvalidOperationException(
                "The query has no KeyColumn, so it describes a single series. Set KeyColumn to scan a population.");
        }

        return await OpenCursorAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Opens a traversal. The returned cursor implements both the single-series and the
    /// grouped contract; which one applies depends on whether a key column is set.
    /// </summary>
    /// <param name="cancellationToken">Cancels opening the connection and executing the query.</param>
    /// <returns>A cursor positioned before the first batch.</returns>
    public async ValueTask<DbSeriesCursor> OpenCursorAsync(CancellationToken cancellationToken = default)
    {
        var connection = _connectionFactory() ?? throw new InvalidOperationException("The connection factory returned null.");
        var cursor = new DbSeriesCursor(this, connection);

        try
        {
            await cursor.OpenAsync(cancellationToken).ConfigureAwait(false);
            return cursor;
        }
        catch
        {
            await cursor.DisposeAsync().ConfigureAwait(false);
            throw;
        }
    }
}
