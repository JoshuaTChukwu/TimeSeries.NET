using TimeSeries.Accumulators;

namespace TimeSeries.Tests.TestUtils;

/// <summary>
/// Drives the accumulators over a series exactly as the estimator's fit loop will:
/// one shared lag window, one push per observation, every accumulator fed from the same
/// row. Until the estimator exists this is where the one-pass contract is exercised.
/// </summary>
public sealed class AccumulatorScan
{
    public AccumulatorScan(int lagDepth, double offset = 0d, bool compensated = false)
    {
        LagDepth = lagDepth;
        Gram = new LagGramAccumulator(lagDepth, offset, compensated);
        Autocovariance = new AutocovarianceAccumulator(lagDepth, offset);
        Moments = new MomentsAccumulator(offset);
    }

    public int LagDepth { get; }

    public LagGramAccumulator Gram { get; }

    public AutocovarianceAccumulator Autocovariance { get; }

    public MomentsAccumulator Moments { get; }

    /// <summary>Scans a whole series.</summary>
    public AccumulatorScan Scan(double[] series)
        => ScanRange(series, 0, series.Length);

    /// <summary>
    /// Scans the rows <c>[start, end)</c> of a series, priming the lag window from the
    /// warm-up prefix immediately before <paramref name="start"/>. The prefix primes only:
    /// it contributes to no accumulator, which is what makes the partitions' states add up
    /// to the single-pass state rather than double-counting the overlap.
    /// </summary>
    public AccumulatorScan ScanRange(double[] series, int start, int end)
    {
        var window = new LagWindow(LagDepth);
        var row = new double[LagDepth + 1];

        for (var t = Math.Max(0, start - LagDepth); t < start; t++)
        {
            window.Push(series[t]);
        }

        for (var t = start; t < end; t++)
        {
            window.Push(series[t]);
            var available = window.CopyRow(row);

            // The classical autocovariances take whatever lags exist; the Gram takes
            // complete rows only.
            Autocovariance.Add(row.AsSpan(0, available));
            Moments.Add(row.AsSpan(0, available));

            if (window.IsFull)
            {
                Gram.Add(row);
            }
        }

        return this;
    }

    public void Merge(AccumulatorScan other)
    {
        Gram.Merge(other.Gram);
        Autocovariance.Merge(other.Autocovariance);
        Moments.Merge(other.Moments);
    }
}
