namespace TimeSeries.Transforms;

/// <summary>
/// Differencing as extension methods on an array, for call sites that read better that way.
/// </summary>
public static class DifferencingExtensions
{
    /// <summary>
    /// Applies differencing of a specified order (d) to a time series.
    /// </summary>
    /// <param name="series">The input time series.</param>
    /// <param name="order">The order of differencing to apply (d).</param>
    /// <returns>A new array containing the differenced series.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="series"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// The order is less than 1, or the series has no more elements than the order.
    /// </exception>
    public static double[] Difference(this double[] series, int order)
    {
        if (series == null)
        {
            throw new ArgumentNullException(nameof(series));
        }

        if (order < 1)
        {
            throw new ArgumentException("Order must be at least 1.");
        }

        if (series.Length <= order)
        {
            throw new ArgumentException("Series must have more elements than the differencing order.");
        }

        return Differencing.Difference(series, new DifferenceSpec(order));
    }

    /// <summary>
    /// Applies <c>(1-B)^d (1-B^s)^D</c> to a time series.
    /// </summary>
    /// <param name="series">The input time series.</param>
    /// <param name="spec">The differencing to apply.</param>
    /// <returns>A new array containing the differenced series.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="series"/> is null.</exception>
    public static double[] Difference(this double[] series, DifferenceSpec spec)
    {
        if (series == null)
        {
            throw new ArgumentNullException(nameof(series));
        }

        return Differencing.Difference(series, spec);
    }
}
