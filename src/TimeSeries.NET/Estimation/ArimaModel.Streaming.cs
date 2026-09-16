using System.Buffers;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
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
    /// Fits one model per key from a population delivered in a single ordered scan, using
    /// <see cref="FitManyOptions.Default"/> — one worker per processor.
    /// </summary>
    /// <param name="source">The population, ordered by key then time.</param>
    /// <param name="cancellationToken">Checked at every batch boundary.</param>
    /// <returns>One result per key, in completion order; see the overload for details.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    public IAsyncEnumerable<KeyedArimaFit> FitManyAsync(IGroupedTimeSeriesSource source, CancellationToken cancellationToken = default)
        => FitManyAsync(source, FitManyOptions.Default, cancellationToken);

    /// <summary>
    /// Fits one model per key from a population delivered in a single ordered scan.
    /// Exactly one series is being read at a time, and at most
    /// <see cref="FitManyOptions.DegreeOfParallelism"/> are being fitted, so working
    /// memory is bounded by the model order and the worker count however many keys there are.
    /// </summary>
    /// <param name="source">The population, ordered by key then time.</param>
    /// <param name="options">Worker count and buffering.</param>
    /// <param name="cancellationToken">Checked at every batch boundary.</param>
    /// <returns>
    /// One result per key, as each series completes — in key order with a single worker,
    /// in completion order otherwise. A key whose fit fails — too short, singular — is
    /// reported with its error rather than aborting the scan; an invalid value in the data
    /// is reported the same way. Ordering violations and source failures propagate.
    /// </returns>
    /// <exception cref="ArgumentNullException">An argument is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The options are inconsistent.</exception>
    public IAsyncEnumerable<KeyedArimaFit> FitManyAsync(
        IGroupedTimeSeriesSource source, FitManyOptions options, CancellationToken cancellationToken = default)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        options.Validate();

        return options.DegreeOfParallelism == 1
            ? FitManySequentialAsync(source, cancellationToken)
            : FitManyParallelAsync(source, options, cancellationToken);
    }

    private async IAsyncEnumerable<KeyedArimaFit> FitManySequentialAsync(
        IGroupedTimeSeriesSource source, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
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

    /// <summary>
    /// Architecture §7.2: a sequential reader is the producer; a bounded channel of keys
    /// feeds a fixed set of workers, each owning one scan at a time; a bounded channel of
    /// results feeds the consumer. Batches are copied into pooled buffers because the
    /// cursor reuses its own.
    /// </summary>
    private async IAsyncEnumerable<KeyedArimaFit> FitManyParallelAsync(
        IGroupedTimeSeriesSource source, FitManyOptions options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var token = linked.Token;

        var dispatch = Channel.CreateBounded<KeyWork>(new BoundedChannelOptions(options.EffectiveKeysInFlight)
        {
            SingleWriter = true,
            SingleReader = false,
        });
        var results = Channel.CreateBounded<KeyedArimaFit>(new BoundedChannelOptions(options.EffectiveKeysInFlight)
        {
            SingleWriter = false,
            SingleReader = true,
        });

        var producer = Task.Run(() => ProduceAsync(source, options, dispatch.Writer, token), CancellationToken.None);
        var workers = new Task[options.DegreeOfParallelism];

        for (var i = 0; i < workers.Length; i++)
        {
            workers[i] = Task.Run(() => WorkAsync(dispatch.Reader, results.Writer, source.RegressorCount, token), CancellationToken.None);
        }

        var completion = CompleteResultsAsync(workers, results.Writer);
        var succeeded = false;

        try
        {
            await foreach (var result in ReadAllAsync(results.Reader, token).ConfigureAwait(false))
            {
                yield return result;
            }

            await producer.ConfigureAwait(false);
            await completion.ConfigureAwait(false);
            succeeded = true;
        }
        finally
        {
            if (!succeeded)
            {
                linked.Cancel();
            }
        }
    }

    private static async Task CompleteResultsAsync(Task[] workers, ChannelWriter<KeyedArimaFit> results)
    {
        try
        {
            await Task.WhenAll(workers).ConfigureAwait(false);
            results.TryComplete();
        }
        catch (Exception exception)
        {
            results.TryComplete(exception);
        }
    }

    private static async Task ProduceAsync(
        IGroupedTimeSeriesSource source, FitManyOptions options, ChannelWriter<KeyWork> dispatch, CancellationToken token)
    {
        Channel<PooledBatch>? batches = null;

        try
        {
            var cursor = await source.OpenAsync(token).ConfigureAwait(false);

            try
            {
                string? key = null;

                while (await cursor.ReadAsync(token).ConfigureAwait(false))
                {
                    var item = cursor.Current;

                    if (batches is null || !string.Equals(item.Key, key, StringComparison.Ordinal))
                    {
                        batches?.Writer.Complete();
                        key = item.Key;
                        batches = Channel.CreateBounded<PooledBatch>(new BoundedChannelOptions(options.BatchesInFlightPerKey)
                        {
                            SingleWriter = true,
                            SingleReader = true,
                        });
                        await dispatch.WriteAsync(new KeyWork(key, batches.Reader), token).ConfigureAwait(false);
                    }

                    await batches.Writer.WriteAsync(PooledBatch.Copy(item.Batch), token).ConfigureAwait(false);
                }

                batches?.Writer.Complete();
                dispatch.Complete();
            }
            finally
            {
                await cursor.DisposeAsync().ConfigureAwait(false);
            }
        }
        catch (Exception exception)
        {
            batches?.Writer.TryComplete(exception);
            dispatch.TryComplete(exception);
            throw;
        }
    }

    private async Task WorkAsync(
        ChannelReader<KeyWork> dispatch, ChannelWriter<KeyedArimaFit> results, int regressorCount, CancellationToken token)
    {
        await foreach (var work in ReadAllAsync(dispatch, token).ConfigureAwait(false))
        {
            var scan = new ArimaScan(Options, regressorCount);
            Exception? failure = null;

            await foreach (var batch in ReadAllAsync(work.Batches, token).ConfigureAwait(false))
            {
                try
                {
                    if (failure is null)
                    {
                        scan.Accept(batch.Values.Span, batch.Exogenous.Span);
                    }
                }
                catch (InvalidSeriesException exception)
                {
                    failure = exception;
                }
                finally
                {
                    batch.Return();
                }
            }

            await results.WriteAsync(Finish(work.Key, scan, failure), token).ConfigureAwait(false);
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

    /// <summary>
    /// Drains a channel. Spelled out because <c>ChannelReader.ReadAllAsync</c> does not
    /// exist in the netstandard2.0 build of System.Threading.Channels.
    /// </summary>
    private static async IAsyncEnumerable<T> ReadAllAsync<T>(
        ChannelReader<T> reader, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        while (await reader.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
        {
            while (reader.TryRead(out var item))
            {
                yield return item;
            }
        }
    }

    private sealed record KeyWork(string Key, ChannelReader<PooledBatch> Batches);

    /// <summary>A batch copied out of a cursor's reusable buffer into pooled arrays.</summary>
    private readonly struct PooledBatch
    {
        private readonly double[] _values;
        private readonly double[] _exogenous;
        private readonly int _count;
        private readonly int _regressors;

        private PooledBatch(double[] values, double[] exogenous, int count, int regressors)
        {
            _values = values;
            _exogenous = exogenous;
            _count = count;
            _regressors = regressors;
        }

        public ReadOnlyMemory<double> Values => _values.AsMemory(0, _count);

        public ReadOnlyMemory<double> Exogenous => _exogenous.AsMemory(0, _count * _regressors);

        public static PooledBatch Copy(SeriesBatch batch)
        {
            var values = ArrayPool<double>.Shared.Rent(Math.Max(batch.Count, 1));
            batch.Values.Span.CopyTo(values);

            var exogenous = batch.RegressorCount == 0
                ? []
                : ArrayPool<double>.Shared.Rent(batch.Count * batch.RegressorCount);
            batch.Exogenous.Span.CopyTo(exogenous);

            return new PooledBatch(values, exogenous, batch.Count, batch.RegressorCount);
        }

        public void Return()
        {
            ArrayPool<double>.Shared.Return(_values);

            if (_exogenous.Length > 0)
            {
                ArrayPool<double>.Shared.Return(_exogenous);
            }
        }
    }
}
