namespace TimeSeries.Accumulators;

/// <summary>
/// Mergeable summary state built from a series in one pass, retaining no data.
/// </summary>
/// <typeparam name="TSelf">The implementing type, so <see cref="Merge"/> stays type-safe.</typeparam>
/// <typeparam name="TResult">The immutable result produced by <see cref="Freeze"/>.</typeparam>
/// <remarks>
/// <para>
/// Two properties carry the architecture. <see cref="Merge"/> is exact, which is what
/// makes a partitioned fit and an incremental fold the same operation seen from two
/// directions — one merges across row ranges, the other across time. And
/// <see cref="Freeze"/> is pure and repeatable, which is what lets order selection score
/// a hundred candidate models against a single accumulator without touching the source
/// again.
/// </para>
/// <para>
/// Implementations are not thread-safe. Use one accumulator per worker and merge when
/// all of them have finished.
/// </para>
/// </remarks>
public interface IAccumulator<TSelf, out TResult>
    where TSelf : IAccumulator<TSelf, TResult>
{
    /// <summary>The number of observations folded in, after any <see cref="Scale"/>.</summary>
    double Count { get; }

    /// <summary>
    /// Folds one observation row into the state.
    /// </summary>
    /// <param name="row">
    /// The lag row ending at the current observation: <c>row[0]</c> is <c>y_t</c> and
    /// <c>row[k]</c> is <c>y_(t-k)</c>. Implementations document how much of it they read.
    /// </param>
    void Add(ReadOnlySpan<double> row);

    /// <summary>
    /// Adds another accumulator's state into this one.
    /// </summary>
    /// <param name="other">The accumulator to absorb. It is not modified.</param>
    /// <remarks>
    /// Exact: merging partial states must give the same answer as accumulating the whole
    /// range in one pass, up to floating-point associativity only.
    /// </remarks>
    void Merge(in TSelf other);

    /// <summary>
    /// Scales the accumulated state by <paramref name="lambda"/>, ageing earlier
    /// observations out at a chosen half-life.
    /// </summary>
    /// <param name="lambda">The forgetting factor, in (0, 1].</param>
    void Scale(double lambda);

    /// <summary>
    /// Produces an immutable snapshot of the current state.
    /// </summary>
    /// <returns>The frozen result. Repeated calls return equal, independent results.</returns>
    TResult Freeze();
}
