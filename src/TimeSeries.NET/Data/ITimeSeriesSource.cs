namespace TimeSeries.Data;

/// <summary>
/// The port through which the estimator reads a series it never holds. Declared in the
/// core and implemented by adapters — <see cref="ArraySource"/> here, database sources in
/// satellite packages — so the estimator depends on an abstraction it owns rather than on
/// anybody's driver.
/// </summary>
public interface ITimeSeriesSource
{
    /// <summary>The number of exogenous regressors each batch carries per observation.</summary>
    int RegressorCount { get; }

    /// <summary>
    /// Opens one traversal of the series. Each call is independent; the cursor must be
    /// disposed, and disposing it releases whatever the traversal held — a database
    /// connection, pooled buffers.
    /// </summary>
    /// <param name="cancellationToken">Cancels the open.</param>
    /// <returns>A cursor positioned before the first batch.</returns>
    ValueTask<ITimeSeriesCursor> OpenAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// One forward-only traversal of a series, in batches. Not thread-safe;
/// <see cref="Current"/> is valid only until the next <see cref="ReadAsync"/>.
/// </summary>
/// <remarks>
/// Batching is why this is not <c>IAsyncEnumerable&lt;double&gt;</c>: an awaited
/// enumerator costs a state-machine hop per item, which at hundreds of millions of rows
/// is seconds of ceremony against a few seconds of arithmetic. Several thousand rows per
/// await amortises that away and leaves the inner loop a synchronous span the JIT can
/// optimise.
/// </remarks>
public interface ITimeSeriesCursor : IAsyncDisposable
{
    /// <summary>
    /// Advances to the next batch.
    /// </summary>
    /// <param name="cancellationToken">Checked at this batch boundary.</param>
    /// <returns>True when <see cref="Current"/> holds a batch; false at the end of the series.</returns>
    ValueTask<bool> ReadAsync(CancellationToken cancellationToken = default);

    /// <summary>The batch read by the last successful <see cref="ReadAsync"/>.</summary>
    SeriesBatch Current { get; }
}

/// <summary>
/// A run of consecutive observations and their regressors. The memory belongs to the
/// cursor and is valid only until it advances.
/// </summary>
public readonly struct SeriesBatch
{
    /// <summary>
    /// Creates a batch over existing memory.
    /// </summary>
    /// <param name="values"><c>Count</c> observations in time order.</param>
    /// <param name="exogenous"><c>Count * RegressorCount</c> values, row-major; empty when there are no regressors.</param>
    /// <param name="regressorCount">Regressors per observation.</param>
    /// <exception cref="ArgumentException">The exogenous memory is not <c>Count * RegressorCount</c> long.</exception>
    public SeriesBatch(ReadOnlyMemory<double> values, ReadOnlyMemory<double> exogenous, int regressorCount)
    {
        if (exogenous.Length != values.Length * regressorCount)
        {
            throw new ArgumentException(
                $"{values.Length} observations with {regressorCount} regressors need {values.Length * regressorCount} " +
                $"exogenous values; {exogenous.Length} supplied.",
                nameof(exogenous));
        }

        Values = values;
        Exogenous = exogenous;
        RegressorCount = regressorCount;
    }

    /// <summary>The observations, in time order.</summary>
    public ReadOnlyMemory<double> Values { get; }

    /// <summary>Regressors, <c>Count</c> rows of <see cref="RegressorCount"/> values, row-major.</summary>
    public ReadOnlyMemory<double> Exogenous { get; }

    /// <summary>Regressors per observation.</summary>
    public int RegressorCount { get; }

    /// <summary>The number of observations in the batch.</summary>
    public int Count => Values.Length;
}
