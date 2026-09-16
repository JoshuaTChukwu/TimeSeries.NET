using System.Runtime.CompilerServices;
using TimeSeries.Data;

namespace TimeSeries;

/// <content>The streaming entry points: the series is never materialised.</content>
public sealed partial class ArimaModel
{
    /// <summary>
    /// Fits the model from a source, reading it exactly once in batches. The cursor is
    /// disposed — and with it any connection — before any numerical work begins.
    /// </summary>
    /// <param name="source">Where the series comes from.</param>
    /// <param name="cancellationToken">Checked at every batch boundary; a cancelled fit stops within one batch.</param>
    /// <returns>The fitted model.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    /// <exception cref="InvalidSeriesException">A value is NaN or infinite; the index is the row ordinal in the scan.</exception>
    /// <exception cref="InsufficientDataException">The series is too short for the model.</exception>
    /// <exception cref="SingularDesignException">The normal equations are singular for a non-constant series.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled.</exception>
    public async ValueTask<ArimaFit> FitAsync(ITimeSeriesSource source, CancellationToken cancellationToken = default)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        var scan = new ArimaScan(Options, source.RegressorCount);
        var cursor = await source.OpenAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            while (await cursor.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                var batch = cursor.Current;
                scan.Accept(batch.Values.Span, batch.Exogenous.Span);
            }
        }
        finally
        {
            await cursor.DisposeAsync().ConfigureAwait(false);
        }

        // From here on: no I/O. The connection has already been released.
        return scan.Solve(new FitWindow(0, scan.RawCount));
    }

    /// <summary>
    /// Fits the model from an asynchronous sequence of values, buffering them into batches
    /// internally. A convenience over <see cref="FitAsync(ITimeSeriesSource, CancellationToken)"/>
    /// for callers who already have an <see cref="IAsyncEnumerable{T}"/>; the batched
    /// cursor is faster at scale.
    /// </summary>
    /// <param name="series">The values, in time order.</param>
    /// <param name="cancellationToken">Checked at every batch boundary.</param>
    /// <returns>The fitted model.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="series"/> is null.</exception>
    public async ValueTask<ArimaFit> FitAsync(IAsyncEnumerable<double> series, CancellationToken cancellationToken = default)
    {
        if (series is null)
        {
            throw new ArgumentNullException(nameof(series));
        }

        var scan = new ArimaScan(Options, regressorCount: 0);
        var buffer = new double[BatchSize];
        var count = 0;

        await foreach (var value in series.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            buffer[count++] = value;

            if (count == BatchSize)
            {
                scan.Accept(buffer, []);
                count = 0;
            }
        }

        if (count > 0)
        {
            scan.Accept(buffer.AsSpan(0, count), []);
        }

        return scan.Solve(new FitWindow(0, scan.RawCount));
    }

    /// <summary>
    /// Fits one model per key from a population delivered in a single ordered scan.
    /// Exactly one series is in progress at a time, so working memory is bounded by the
    /// model order however many keys there are.
    /// </summary>
    /// <param name="source">The population, ordered by key then time.</param>
    /// <param name="cancellationToken">Checked at every batch boundary.</param>
    /// <returns>
    /// One result per key, in scan order, as each series completes. A key whose fit fails
    /// — too short, singular — is reported with its error rather than aborting the scan;
    /// an invalid value in the data is reported the same way. Ordering violations and
    /// source failures propagate.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    public async IAsyncEnumerable<KeyedArimaFit> FitManyAsync(
        IGroupedTimeSeriesSource source,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        var cursor = await source.OpenAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            string? key = null;
            ArimaScan? scan = null;
            Exception? failure = null;

            while (await cursor.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                var item = cursor.Current;

                if (scan is null || !string.Equals(item.Key, key, StringComparison.Ordinal))
                {
                    if (scan is not null)
                    {
                        yield return Finish(key!, scan, failure);
                    }

                    key = item.Key;
                    scan = new ArimaScan(Options, source.RegressorCount);
                    failure = null;
                }

                if (failure is not null)
                {
                    // This key is already known bad; its remaining rows are skipped.
                    continue;
                }

                try
                {
                    scan.Accept(item.Batch.Values.Span, item.Batch.Exogenous.Span);
                }
                catch (InvalidSeriesException exception)
                {
                    failure = exception;
                }
            }

            if (scan is not null)
            {
                yield return Finish(key!, scan, failure);
            }
        }
        finally
        {
            await cursor.DisposeAsync().ConfigureAwait(false);
        }
    }

    private static KeyedArimaFit Finish(string key, ArimaScan scan, Exception? failure)
    {
        if (failure is not null)
        {
            return new KeyedArimaFit(key, null, failure, scan.RawCount);
        }

        try
        {
            return new KeyedArimaFit(key, scan.Solve(new FitWindow(0, scan.RawCount)), null, scan.RawCount);
        }
        catch (InsufficientDataException exception)
        {
            return new KeyedArimaFit(key, null, exception, scan.RawCount);
        }
        catch (SingularDesignException exception)
        {
            return new KeyedArimaFit(key, null, exception, scan.RawCount);
        }
    }
}
