namespace TimeSeries.Statistics;

/// <summary>
/// Point-forecast and interval accuracy metrics.
/// </summary>
/// <remarks>
/// <para>
/// Every metric takes actuals and forecasts of equal, non-zero length and refuses
/// anything else. The percentage metrics return percentages (a MAPE of 5 means 5%), and
/// MAPE throws on a zero actual rather than dropping the point — a metric computed over a
/// subset the caller did not choose is not the metric they asked for.
/// </para>
/// <para>
/// MAPE is asymmetric and undefined at zero, which is why MASE is offered alongside it:
/// scale-free, always defined for a non-constant training series, and comparable across
/// series of different scales.
/// </para>
/// </remarks>
public static class ErrorMetrics
{
    /// <summary>Mean absolute error, <c>mean |a - f|</c>, in the units of the series.</summary>
    /// <param name="actual">Observed values.</param>
    /// <param name="forecast">Forecast values, aligned with <paramref name="actual"/>.</param>
    /// <returns>The mean absolute error.</returns>
    /// <exception cref="ArgumentException">The spans are empty or differ in length.</exception>
    public static double Mae(ReadOnlySpan<double> actual, ReadOnlySpan<double> forecast)
    {
        Check(actual, forecast);
        var sum = 0d;

        for (var i = 0; i < actual.Length; i++)
        {
            sum += Math.Abs(actual[i] - forecast[i]);
        }

        return sum / actual.Length;
    }

    /// <summary>Mean squared error, <c>mean (a - f)^2</c>, in squared units.</summary>
    /// <param name="actual">Observed values.</param>
    /// <param name="forecast">Forecast values, aligned with <paramref name="actual"/>.</param>
    /// <returns>The mean squared error.</returns>
    /// <exception cref="ArgumentException">The spans are empty or differ in length.</exception>
    public static double Mse(ReadOnlySpan<double> actual, ReadOnlySpan<double> forecast)
    {
        Check(actual, forecast);
        var sum = 0d;

        for (var i = 0; i < actual.Length; i++)
        {
            var error = actual[i] - forecast[i];
            sum += error * error;
        }

        return sum / actual.Length;
    }

    /// <summary>Root mean squared error, <c>sqrt(MSE)</c>, in the units of the series.</summary>
    /// <param name="actual">Observed values.</param>
    /// <param name="forecast">Forecast values, aligned with <paramref name="actual"/>.</param>
    /// <returns>The root mean squared error.</returns>
    /// <exception cref="ArgumentException">The spans are empty or differ in length.</exception>
    public static double Rmse(ReadOnlySpan<double> actual, ReadOnlySpan<double> forecast)
        => Math.Sqrt(Mse(actual, forecast));

    /// <summary>
    /// Mean absolute percentage error, <c>100 * mean |a - f| / |a|</c>.
    /// </summary>
    /// <param name="actual">Observed values. None may be zero.</param>
    /// <param name="forecast">Forecast values, aligned with <paramref name="actual"/>.</param>
    /// <returns>The error as a percentage.</returns>
    /// <exception cref="ArgumentException">
    /// The spans are empty or differ in length, or an actual is exactly zero — the message
    /// names its index. The point is never silently dropped.
    /// </exception>
    public static double Mape(ReadOnlySpan<double> actual, ReadOnlySpan<double> forecast)
    {
        Check(actual, forecast);
        var sum = 0d;

        for (var i = 0; i < actual.Length; i++)
        {
            if (actual[i] == 0d)
            {
                throw new ArgumentException(
                    $"MAPE is undefined when an actual is zero, and actual[{i}] is. Use sMAPE or MASE " +
                    "for series that touch zero.",
                    nameof(actual));
            }

            sum += Math.Abs((actual[i] - forecast[i]) / actual[i]);
        }

        return 100d * sum / actual.Length;
    }

    /// <summary>
    /// Symmetric mean absolute percentage error,
    /// <c>200 * mean |a - f| / (|a| + |f|)</c>, in [0, 200].
    /// </summary>
    /// <param name="actual">Observed values.</param>
    /// <param name="forecast">Forecast values, aligned with <paramref name="actual"/>.</param>
    /// <returns>The error as a percentage. A point where both values are zero contributes 0.</returns>
    /// <exception cref="ArgumentException">The spans are empty or differ in length.</exception>
    public static double Smape(ReadOnlySpan<double> actual, ReadOnlySpan<double> forecast)
    {
        Check(actual, forecast);
        var sum = 0d;

        for (var i = 0; i < actual.Length; i++)
        {
            var denominator = Math.Abs(actual[i]) + Math.Abs(forecast[i]);

            if (denominator > 0d)
            {
                sum += Math.Abs(actual[i] - forecast[i]) / denominator;
            }
        }

        return 200d * sum / actual.Length;
    }

    /// <summary>
    /// Mean absolute scaled error: MAE relative to the in-sample MAE of the seasonal naive
    /// forecast <c>y_t = y_(t-m)</c> on the training series. Below 1 beats the naive
    /// forecast; above 1 does not.
    /// </summary>
    /// <param name="actual">Observed values over the forecast window.</param>
    /// <param name="forecast">Forecast values, aligned with <paramref name="actual"/>.</param>
    /// <param name="training">The series the forecast was fitted on, in time order.</param>
    /// <param name="period">The seasonal period <c>m</c>; 1 for the plain naive forecast.</param>
    /// <returns>The scaled error, dimensionless.</returns>
    /// <exception cref="ArgumentException">
    /// The spans are empty or differ in length; the training series is no longer than
    /// <paramref name="period"/>; or the naive forecast is exact on the training series,
    /// leaving nothing to scale by.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="period"/> is below 1.</exception>
    public static double Mase(
        ReadOnlySpan<double> actual, ReadOnlySpan<double> forecast, ReadOnlySpan<double> training, int period = 1)
    {
        Check(actual, forecast);

        if (period < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(period), period, "Period must be 1 or greater.");
        }

        if (training.Length <= period)
        {
            throw new ArgumentException(
                $"The training series must be longer than the period ({period}) to form a naive forecast; " +
                $"it has {training.Length} values.",
                nameof(training));
        }

        var scale = 0d;

        for (var t = period; t < training.Length; t++)
        {
            scale += Math.Abs(training[t] - training[t - period]);
        }

        scale /= training.Length - period;

        if (scale == 0d)
        {
            throw new ArgumentException(
                "The seasonal naive forecast is exact on the training series, so MASE has a zero " +
                "denominator. The training series is constant at the given period.",
                nameof(training));
        }

        return Mae(actual, forecast) / scale;
    }

    /// <summary>
    /// The fraction of actuals that fell inside <c>[lower, upper]</c>. For a nominal 95%
    /// interval this should be near 0.95; well below means the intervals are too narrow.
    /// </summary>
    /// <param name="actual">Observed values.</param>
    /// <param name="lower">Lower interval bounds, aligned with <paramref name="actual"/>.</param>
    /// <param name="upper">Upper interval bounds, aligned with <paramref name="actual"/>.</param>
    /// <returns>The empirical coverage, in [0, 1].</returns>
    /// <exception cref="ArgumentException">The spans are empty or differ in length.</exception>
    public static double Coverage(ReadOnlySpan<double> actual, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper)
    {
        SeriesGuard.SameNonEmptyLength(actual, nameof(actual), lower, nameof(lower));
        SeriesGuard.SameNonEmptyLength(actual, nameof(actual), upper, nameof(upper));

        var inside = 0;

        for (var i = 0; i < actual.Length; i++)
        {
            if (actual[i] >= lower[i] && actual[i] <= upper[i])
            {
                inside++;
            }
        }

        return (double)inside / actual.Length;
    }

    private static void Check(ReadOnlySpan<double> actual, ReadOnlySpan<double> forecast)
        => SeriesGuard.SameNonEmptyLength(actual, nameof(actual), forecast, nameof(forecast));
}
