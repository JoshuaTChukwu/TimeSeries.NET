namespace TimeSeries.Transforms;

/// <summary>
/// The bounded state needed to undo a <see cref="DifferenceSpec"/>: the tail each
/// differencing stage consumed, and nothing else.
/// </summary>
/// <remarks>
/// <para>
/// This replaces passing the original series to an inverse-difference call. It is
/// <c>d + D * s</c> doubles — bounded by the model, not by the series — which is why a
/// model fitted over hundreds of millions of rows still serialises to a few hundred
/// bytes and can be stored, shipped, and forecast from elsewhere.
/// </para>
/// <para>
/// Instances are immutable. <see cref="Integrate(ReadOnlySpan{double})"/> works on a copy
/// of the tail, so the same state can seed any number of integrations and always
/// produces the same answer.
/// </para>
/// </remarks>
public sealed class IntegrationState
{
    private readonly double[] _tail;

    /// <summary>
    /// Creates an integration state from a previously captured tail.
    /// </summary>
    /// <param name="spec">The differencing this state inverts.</param>
    /// <param name="tail">
    /// The stage windows laid out end to end in application order — seasonal stages
    /// first, then non-seasonal — each in chronological order. Length must be
    /// <see cref="DifferenceSpec.WarmupLength"/>. The span is copied.
    /// </param>
    /// <exception cref="ArgumentException">
    /// <paramref name="tail"/> has the wrong length for <paramref name="spec"/>.
    /// </exception>
    public IntegrationState(DifferenceSpec spec, ReadOnlySpan<double> tail)
    {
        if (tail.Length != spec.WarmupLength)
        {
            throw new ArgumentException(
                $"Integration state for {spec} needs exactly {spec.WarmupLength} values; " +
                $"{tail.Length} supplied.",
                nameof(tail));
        }

        Spec = spec;
        _tail = tail.ToArray();
    }

    /// <summary>The differencing this state inverts.</summary>
    public DifferenceSpec Spec { get; }

    /// <summary>
    /// The captured stage windows, laid out end to end in application order. Exposed so
    /// the state can be serialised alongside a fitted model.
    /// </summary>
    public ReadOnlyMemory<double> Tail => _tail;

    /// <summary>
    /// Captures the state that reconstructs a series from the start of its differenced
    /// form — the inverse of differencing that same series.
    /// </summary>
    /// <param name="series">The undifferenced series.</param>
    /// <param name="spec">The differencing to invert.</param>
    /// <returns>State seeded from the first <c>d + D * s</c> observations.</returns>
    /// <exception cref="ArgumentException">
    /// <paramref name="series"/> is shorter than <see cref="DifferenceSpec.WarmupLength"/>.
    /// </exception>
    public static IntegrationState FromStart(ReadOnlySpan<double> series, DifferenceSpec spec)
        => Capture(series, spec, spec.WarmupLength);

    /// <summary>
    /// Captures the state that continues a series past its final observation, which is
    /// what a forecast is integrated through.
    /// </summary>
    /// <param name="series">The undifferenced series.</param>
    /// <param name="spec">The differencing to invert.</param>
    /// <returns>State seeded from the last <c>d + D * s</c> observations.</returns>
    /// <exception cref="ArgumentException">
    /// <paramref name="series"/> is shorter than <see cref="DifferenceSpec.WarmupLength"/>.
    /// </exception>
    public static IntegrationState FromEnd(ReadOnlySpan<double> series, DifferenceSpec spec)
        => Capture(series, spec, series.Length);

    /// <summary>
    /// Integrates a differenced series back to the original scale.
    /// </summary>
    /// <param name="differenced">Values on the differenced scale, in time order.</param>
    /// <returns>A new array of the same length, on the original scale.</returns>
    public double[] Integrate(ReadOnlySpan<double> differenced)
    {
        var result = new double[differenced.Length];
        Integrate(differenced, result);
        return result;
    }

    /// <summary>
    /// Integrates a differenced series back to the original scale, into a caller-supplied
    /// buffer.
    /// </summary>
    /// <param name="differenced">Values on the differenced scale, in time order.</param>
    /// <param name="output">
    /// Destination, at least as long as <paramref name="differenced"/>. May be the same
    /// buffer.
    /// </param>
    /// <returns>The number of values written, always the length of <paramref name="differenced"/>.</returns>
    /// <exception cref="ArgumentException">
    /// <paramref name="output"/> is shorter than <paramref name="differenced"/>.
    /// </exception>
    public int Integrate(ReadOnlySpan<double> differenced, Span<double> output)
    {
        if (output.Length < differenced.Length)
        {
            throw new ArgumentException(
                $"Output span must be at least as long as the differenced series: " +
                $"{differenced.Length} values supplied, {output.Length} available.",
                nameof(output));
        }

        var count = differenced.Length;
        differenced.CopyTo(output);

        // A working copy, so the state itself stays immutable and reusable.
        Span<double> tail = _tail.Length <= 128 ? stackalloc double[_tail.Length] : new double[_tail.Length];
        _tail.AsSpan().CopyTo(tail);

        // Stages were applied seasonal-first; undo them in the reverse order.
        var offset = _tail.Length;

        for (var stage = StageCount - 1; stage >= 0; stage--)
        {
            var lag = LagOfStage(stage);
            offset -= lag;

            var window = tail.Slice(offset, lag);
            var head = 0;

            for (var t = 0; t < count; t++)
            {
                var value = output[t] + window[head];
                output[t] = value;
                window[head] = value;
                head = head + 1 == lag ? 0 : head + 1;
            }
        }

        return count;
    }

    private int StageCount => Spec.SeasonalOrder + Spec.Order;

    private int LagOfStage(int index) => index < Spec.SeasonalOrder ? Spec.Period : 1;

    private static IntegrationState Capture(
        ReadOnlySpan<double> series, DifferenceSpec spec, int take)
    {
        if (series.Length < spec.WarmupLength)
        {
            throw new ArgumentException(
                $"Differencing {spec} needs at least {spec.WarmupLength} observations to seed " +
                $"integration; the series has {series.Length}.",
                nameof(series));
        }

        var transform = new DifferenceTransform(spec);

        // Fed in chunks so that seeding from the end of a long series costs a fixed
        // scratch buffer rather than a second copy of the series. The output is
        // discarded: only the stage windows left behind are wanted.
        const int ChunkSize = 1024;
        Span<double> discard = stackalloc double[ChunkSize];

        for (var start = 0; start < take; start += ChunkSize)
        {
            var length = Math.Min(ChunkSize, take - start);
            transform.Transform(series.Slice(start, length), discard);
        }

        return transform.CaptureState();
    }
}
