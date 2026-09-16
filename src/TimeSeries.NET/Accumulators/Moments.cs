namespace TimeSeries.Accumulators;

/// <summary>Frozen first and second moments of a series, plus its range.</summary>
public sealed class Moments
{
    internal Moments(
        double count, double mean, double variance, double sampleVariance,
        double minimum, double maximum)
    {
        Count = count;
        Mean = mean;
        Variance = variance;
        SampleVariance = sampleVariance;
        Minimum = minimum;
        Maximum = maximum;
    }

    /// <summary>The number of observations, after any forgetting factor.</summary>
    public double Count { get; }

    /// <summary>The arithmetic mean, on the original scale.</summary>
    public double Mean { get; }

    /// <summary>The divide-by-n variance, consistent with the autocovariance estimator.</summary>
    public double Variance { get; }

    /// <summary>The divide-by-(n-1) variance, for reporting. NaN below two observations.</summary>
    public double SampleVariance { get; }

    /// <summary>The smallest observation, on the original scale.</summary>
    public double Minimum { get; }

    /// <summary>The largest observation, on the original scale.</summary>
    public double Maximum { get; }

    /// <summary>The square root of <see cref="Variance"/>.</summary>
    public double StandardDeviation => Math.Sqrt(Variance);
}
