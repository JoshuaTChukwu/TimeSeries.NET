namespace TimeSeries.Accumulators;

/// <summary>
/// Frozen classical autocovariances <c>gamma-hat_k</c> and the autocorrelations derived
/// from them.
/// </summary>
/// <remarks>
/// The divide-by-n (biased) estimator, deliberately. Dividing by <c>n</c> rather than by
/// <c>n - k</c> is what makes the sequence positive semi-definite, and that is what keeps
/// Durbin-Levinson stable when it recurses through it for the partial autocorrelations.
/// </remarks>
public sealed class Autocovariances
{
    private readonly double[] _gamma;
    private readonly double[] _acf;

    internal Autocovariances(int maxLag, double[] gamma, double[] acf, double mean, double count)
    {
        MaxLag = maxLag;
        _gamma = gamma;
        _acf = acf;
        Mean = mean;
        Count = count;
    }

    /// <summary>The deepest lag estimated, <c>L</c>.</summary>
    public int MaxLag { get; }

    /// <summary>The number of observations the estimates are built from.</summary>
    public double Count { get; }

    /// <summary>The series mean, on the original scale.</summary>
    public double Mean { get; }

    /// <summary>
    /// Autocovariances for lags 0 through <see cref="MaxLag"/>. Index 0 is the variance.
    /// </summary>
    public ReadOnlyMemory<double> Gamma => _gamma;

    /// <summary>
    /// Autocorrelations <c>rho-hat_k = gamma-hat_k / gamma-hat_0</c> for lags 0 through
    /// <see cref="MaxLag"/>. For a constant series, where the variance is zero, lag 0 is
    /// 1 and every other lag is 0.
    /// </summary>
    public ReadOnlyMemory<double> Acf => _acf;

    /// <summary>The series variance, <c>gamma-hat_0</c>.</summary>
    public double Variance => _gamma[0];
}
