namespace TimeSeries;

/// <summary>
/// How <see cref="ArimaModel.FitManyAsync(Data.IGroupedTimeSeriesSource, FitManyOptions, CancellationToken)"/>
/// spreads a population over workers.
/// </summary>
/// <remarks>
/// The reader stays sequential — the ordering by key then time is what keeps memory
/// bounded — and hands each key's batches to one of a fixed set of workers through a
/// bounded channel. Back-pressure comes from the bounds: the scan cannot outrun the
/// fitters and pile up buffered series. Working memory is
/// <c>workers × (accumulator state + BatchesInFlightPerKey × batch)</c>, independent of the
/// number of keys. Results arrive in completion order, not key order.
/// </remarks>
public sealed record FitManyOptions
{
    /// <summary>One worker per processor, four batches in flight per key.</summary>
    public static FitManyOptions Default { get; } = new();

    /// <summary>A single worker: results in key order, on the caller's thread.</summary>
    public static FitManyOptions Sequential { get; } = new() { DegreeOfParallelism = 1 };

    /// <summary>Worker tasks fitting keys concurrently. Default <see cref="Environment.ProcessorCount"/>; 1 is sequential.</summary>
    public int DegreeOfParallelism { get; init; } = Environment.ProcessorCount;

    /// <summary>Copied batches a worker may have queued for the key it is fitting. Default 4.</summary>
    public int BatchesInFlightPerKey { get; init; } = 4;

    /// <summary>Keys the reader may run ahead of the workers by. Zero, the default, means twice the workers.</summary>
    public int KeysInFlight { get; init; }

    /// <summary>Checks the options.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A bound is below its minimum.</exception>
    public void Validate()
    {
        if (DegreeOfParallelism < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(DegreeOfParallelism), DegreeOfParallelism, "At least one worker is required.");
        }

        if (BatchesInFlightPerKey < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(BatchesInFlightPerKey), BatchesInFlightPerKey, "At least one batch in flight is required.");
        }

        if (KeysInFlight < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(KeysInFlight), KeysInFlight, "Keys in flight must be zero (automatic) or greater.");
        }
    }

    internal int EffectiveKeysInFlight => KeysInFlight > 0 ? KeysInFlight : 2 * DegreeOfParallelism;
}
