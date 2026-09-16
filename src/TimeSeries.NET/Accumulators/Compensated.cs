namespace TimeSeries.Accumulators;

/// <summary>
/// Neumaier compensated summation, used where an accumulator opts into it.
/// </summary>
/// <remarks>
/// Roughly twice the flops of a plain add, on a path that is bound by I/O rather than by
/// arithmetic. It matters over the 10^8 additions a long series produces, where the drift
/// of ordinary summation is no longer negligible against the quantities being estimated.
/// </remarks>
internal static class Compensated
{
    /// <summary>
    /// Adds <paramref name="value"/> to a running sum, routing the lost low-order bits
    /// into <paramref name="compensation"/>.
    /// </summary>
    internal static void Add(ref double sum, ref double compensation, double value)
    {
        var total = sum + value;

        // Whichever operand is larger keeps its bits; the smaller one loses them, and
        // the difference is exactly what the compensation term recovers.
        compensation += Math.Abs(sum) >= Math.Abs(value)
            ? (sum - total) + value
            : (value - total) + sum;

        sum = total;
    }
}
