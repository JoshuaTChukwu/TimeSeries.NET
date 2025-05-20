namespace TimeSeries.Core.Math;
public static class DifferencingExtensions
{
    /// <summary>
    /// Applies differencing of a specified order (d) to a time series.
    /// </summary>
    /// <param name="series">The input time series.</param>
    /// <param name="order">The order of differencing to apply (d).</param>
    /// <returns>A new array containing the differenced series.</returns>
    public static double[] Difference(this double[] series, int order)
    {
        if (series == null)
            throw new ArgumentNullException(nameof(series));
        if (order < 1)
            throw new ArgumentException("Order must be at least 1.");
        if (series.Length <= order)
            throw new ArgumentException("Series must have more elements than the differencing order.");

        double[] current = series;

        for (int d = 0; d < order; d++)
        {
            double[] result = new double[current.Length - 1];
            for (int i = 1; i < current.Length; i++)
            {
                result[i - 1] = current[i] - current[i - 1];
            }
            current = result;
        }

        return current;
    }
}