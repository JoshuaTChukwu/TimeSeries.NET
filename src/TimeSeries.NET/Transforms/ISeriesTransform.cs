namespace TimeSeries.Transforms;

/// <summary>
/// A stateful, streaming transform over a series. Consumes batches of values and
/// produces transformed values, holding only the warm-up window rather than the series.
/// </summary>
/// <remarks>
/// Implementations are not thread-safe: one instance belongs to one traversal. Values
/// must arrive in time order, and state persists across calls to
/// <see cref="Transform"/> so a batch boundary is invisible to the result.
/// </remarks>
public interface ISeriesTransform
{
    /// <summary>
    /// Observations consumed before the first output value is produced.
    /// </summary>
    /// <remarks>
    /// A partitioned reader sizes its overlap prefix from this number, so reporting it
    /// correctly is the transform's responsibility rather than the reader's to guess.
    /// </remarks>
    int WarmupLength { get; }

    /// <summary>
    /// Transforms one batch.
    /// </summary>
    /// <param name="input">The batch of source values, in time order.</param>
    /// <param name="output">
    /// Destination for transformed values. Must be at least as long as
    /// <paramref name="input"/>, since a transform never produces more values than it
    /// consumes.
    /// </param>
    /// <returns>
    /// The number of values written to <paramref name="output"/>. Fewer than
    /// <paramref name="input"/> holds only while warm-up is still being consumed.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// <paramref name="output"/> is shorter than <paramref name="input"/>.
    /// </exception>
    int Transform(ReadOnlySpan<double> input, Span<double> output);

    /// <summary>
    /// Discards all state, returning the transform to its pre-warm-up condition so it
    /// can be reused for another series.
    /// </summary>
    void Reset();
}
