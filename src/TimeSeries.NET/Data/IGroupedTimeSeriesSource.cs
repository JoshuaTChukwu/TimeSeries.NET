namespace TimeSeries.Data;

/// <summary>
/// A population of series delivered in one ordered scan — ordered by key, then time — so
/// that exactly one series is in progress at any moment and working memory is bounded by
/// the model order rather than by the number of keys.
/// </summary>
/// <remarks>
/// The ordering is the source's responsibility to guarantee and to validate: a key that
/// reappears after another key has begun would otherwise be fitted as a mixture, and the
/// source must throw <see cref="SeriesOrderException"/> instead. In a database this means
/// a composite index on <c>(key, time)</c>; without one the server sorts on disk and the
/// problem moves rather than disappears.
/// </remarks>
public interface IGroupedTimeSeriesSource
{
    /// <summary>The number of exogenous regressors each batch carries per observation.</summary>
    int RegressorCount { get; }

    /// <summary>Opens one traversal of the whole population.</summary>
    /// <param name="cancellationToken">Cancels the open.</param>
    /// <returns>A cursor positioned before the first batch.</returns>
    ValueTask<IGroupedTimeSeriesCursor> OpenAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// One traversal of a keyed population. A batch never spans two keys, so a change of
/// <see cref="KeyedSeriesBatch.Key"/> between consecutive batches is the boundary between
/// two series.
/// </summary>
public interface IGroupedTimeSeriesCursor : IAsyncDisposable
{
    /// <summary>Advances to the next batch.</summary>
    /// <param name="cancellationToken">Checked at this batch boundary.</param>
    /// <returns>True when <see cref="Current"/> holds a batch; false at the end of the scan.</returns>
    ValueTask<bool> ReadAsync(CancellationToken cancellationToken = default);

    /// <summary>The batch read by the last successful <see cref="ReadAsync"/>, with its key.</summary>
    KeyedSeriesBatch Current { get; }
}

/// <summary>A batch of one series in a keyed population.</summary>
public readonly struct KeyedSeriesBatch
{
    /// <summary>Creates a keyed batch.</summary>
    /// <param name="key">The series' key, as text.</param>
    /// <param name="batch">The observations.</param>
    /// <exception cref="ArgumentNullException"><paramref name="key"/> is null.</exception>
    public KeyedSeriesBatch(string key, SeriesBatch batch)
    {
        Key = key ?? throw new ArgumentNullException(nameof(key));
        Batch = batch;
    }

    /// <summary>The series' key.</summary>
    public string Key { get; }

    /// <summary>The observations belonging to <see cref="Key"/>.</summary>
    public SeriesBatch Batch { get; }
}

/// <summary>
/// The outcome for one key of a population fit: a model, or the reason there is none.
/// One short series among a million must not abort the scan, so failures are returned
/// rather than thrown.
/// </summary>
public sealed class KeyedArimaFit
{
    internal KeyedArimaFit(string key, ArimaFit? fit, Exception? error, long observationCount)
    {
        Key = key;
        Fit = fit;
        Error = error;
        ObservationCount = observationCount;
    }

    /// <summary>The series' key.</summary>
    public string Key { get; }

    /// <summary>The fitted model, or null when the fit failed.</summary>
    public ArimaFit? Fit { get; }

    /// <summary>
    /// Why the fit failed — typically <see cref="InsufficientDataException"/> for a short
    /// series or <see cref="SingularDesignException"/> — or null on success.
    /// </summary>
    public Exception? Error { get; }

    /// <summary>Raw observations read for this key.</summary>
    public long ObservationCount { get; }

    /// <summary>True when <see cref="Fit"/> is present.</summary>
    public bool Succeeded => Fit is not null;
}
