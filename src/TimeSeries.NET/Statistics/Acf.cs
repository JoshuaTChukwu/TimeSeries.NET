using TimeSeries.Accumulators;

namespace TimeSeries.Statistics;

/// <summary>
/// The sample autocorrelation function and its white-noise bounds.
/// </summary>
/// <remarks>
/// A thin entry point over <see cref="AutocovarianceAccumulator"/>: the same one-pass
/// accumulation the estimator runs, applied to an array. There is no second
/// implementation of the estimator here.
/// </remarks>
public static class Acf
{
    /// <summary>
    /// Estimates autocovariances and autocorrelations for lags 0 through <paramref name="maxLag"/>.
    /// </summary>
    /// <param name="series">The series, in time order.</param>
    /// <param name="maxLag">The deepest lag to estimate. Must be less than the series length.</param>
    /// <returns>The divide-by-n autocovariances, autocorrelations, mean and count.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is negative.</exception>
    /// <exception cref="ArgumentException">The series is no longer than <paramref name="maxLag"/>.</exception>
    /// <exception cref="InvalidSeriesException">The series contains NaN or an infinity.</exception>
    public static Autocovariances Compute(ReadOnlySpan<double> series, int maxLag)
    {
        if (maxLag < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLag), maxLag, "Maximum lag must be zero or greater.");
        }

        if (series.Length <= maxLag)
        {
            throw new ArgumentException(
                $"Estimating {maxLag} lags needs more than {maxLag} observations; the series has {series.Length}.",
                nameof(series));
        }

        SeriesGuard.Finite(series, nameof(series));

        // The first observation is the offset, per the numerical contract: it costs
        // nothing and keeps a large level from swamping the deviations.
        var accumulator = new AutocovarianceAccumulator(maxLag, offset: series[0]);
        var window = new LagWindow(maxLag);
        Span<double> row = maxLag < 256 ? stackalloc double[maxLag + 1] : new double[maxLag + 1];

        for (var t = 0; t < series.Length; t++)
        {
            window.Push(series[t]);
            var available = window.CopyRow(row);
            accumulator.Add(row.Slice(0, available));
        }

        return accumulator.Freeze();
    }

    /// <summary>
    /// The half-width of the band inside which a sample autocorrelation of white noise
    /// falls with the given confidence: <c>z / sqrt(n)</c>.
    /// </summary>
    /// <param name="count">The number of observations the autocorrelations were estimated from.</param>
    /// <param name="confidence">The confidence level, strictly inside (0, 1). Default 0.95.</param>
    /// <returns>The bound; a lag whose autocorrelation lies outside <c>±bound</c> is significant.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="count"/> is not positive, or <paramref name="confidence"/> is outside (0, 1).
    /// </exception>
    public static double WhiteNoiseBound(double count, double confidence = 0.95)
    {
        if (!(count > 0d))
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, "Count must be positive.");
        }

        return NormalDistribution.TwoSidedCriticalValue(confidence) / Math.Sqrt(count);
    }
}
