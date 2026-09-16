using TimeSeries.Accumulators;
using TimeSeries.Solvers;

namespace TimeSeries.Statistics;

/// <summary>
/// The sample partial autocorrelation function, by Durbin-Levinson over the classical
/// autocovariances.
/// </summary>
public static class Pacf
{
    /// <summary>
    /// Estimates partial autocorrelations for lags 1 through <paramref name="maxLag"/>.
    /// </summary>
    /// <param name="series">The series, in time order.</param>
    /// <param name="maxLag">The deepest lag to estimate. Must be less than the series length.</param>
    /// <returns><c>phi_kk</c> for <c>k = 1 .. maxLag</c> at indices <c>0 .. maxLag-1</c>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is negative.</exception>
    /// <exception cref="ArgumentException">The series is no longer than <paramref name="maxLag"/>.</exception>
    /// <exception cref="InvalidSeriesException">The series contains NaN or an infinity.</exception>
    public static double[] Compute(ReadOnlySpan<double> series, int maxLag)
        => FromAutocovariances(Acf.Compute(series, maxLag), maxLag);

    /// <summary>
    /// Derives partial autocorrelations from already-estimated autocovariances.
    /// </summary>
    /// <param name="autocovariances">Autocovariances estimated to at least <paramref name="maxLag"/>.</param>
    /// <param name="maxLag">The deepest lag to compute.</param>
    /// <returns>
    /// <c>phi_kk</c> for <c>k = 1 .. maxLag</c>. For a constant series every value is 0.
    /// Should the recursion become degenerate at some lag — possible only on a sequence
    /// that is not positive semi-definite, which the library's estimator never produces —
    /// the lags beyond it are NaN.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="autocovariances"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="maxLag"/> is negative or exceeds the estimated depth.
    /// </exception>
    public static double[] FromAutocovariances(Autocovariances autocovariances, int maxLag)
    {
        if (autocovariances is null)
        {
            throw new ArgumentNullException(nameof(autocovariances));
        }

        if (maxLag < 0 || maxLag > autocovariances.MaxLag)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maxLag), maxLag,
                $"Maximum lag must lie in [0, {autocovariances.MaxLag}], the depth the autocovariances were estimated to.");
        }

        var pacf = new double[maxLag];

        if (maxLag == 0)
        {
            return pacf;
        }

        if (!(autocovariances.Variance > 0d))
        {
            // A constant series: nothing is partially correlated with anything.
            return pacf;
        }

        if (!DurbinLevinson.TryPacf(autocovariances.Gamma.Span, maxLag, pacf, out var computed))
        {
            for (var k = computed; k < maxLag; k++)
            {
                pacf[k] = double.NaN;
            }
        }

        return pacf;
    }
}
