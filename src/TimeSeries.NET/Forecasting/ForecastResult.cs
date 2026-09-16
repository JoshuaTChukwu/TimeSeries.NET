using TimeSeries.Statistics;

namespace TimeSeries;

/// <summary>
/// A forecast path with its prediction intervals. Intervals are never optional: an
/// ARIMA point forecast converges to its drift line within a few steps, after which the
/// interval is the entire content of the forecast.
/// </summary>
/// <remarks>
/// The path is a joint distribution, not a list of independent numbers. Errors at
/// different steps share innovations and are correlated, so aggregating over steps —
/// total sales next quarter, cumulative demand — must use
/// <see cref="Sum(int, int)"/>, which carries the covariance, rather than adding
/// per-step variances.
/// </remarks>
public sealed class ForecastResult
{
    private readonly double[] _mean;
    private readonly double[] _standardError;
    private readonly double[] _lower;
    private readonly double[] _upper;
    private readonly double[] _psi;
    private readonly string[] _warnings;

    internal ForecastResult(
        double[] mean,
        double[] standardError,
        double[] psi,
        double innovationVariance,
        double confidence,
        long origin,
        string[] warnings)
    {
        _mean = mean;
        _standardError = standardError;
        _psi = psi;
        InnovationVariance = innovationVariance;
        Confidence = confidence;
        Origin = origin;
        _warnings = warnings;

        CriticalValue = NormalDistribution.TwoSidedCriticalValue(confidence);
        _lower = new double[mean.Length];
        _upper = new double[mean.Length];

        for (var h = 0; h < mean.Length; h++)
        {
            var halfWidth = CriticalValue * standardError[h];
            _lower[h] = mean[h] - halfWidth;
            _upper[h] = mean[h] + halfWidth;
        }
    }

    /// <summary>The number of steps forecast.</summary>
    public int Steps => _mean.Length;

    /// <summary>The row index the forecast starts from: step 1 is row <c>Origin</c>.</summary>
    public long Origin { get; }

    /// <summary>The confidence level of <see cref="Lower"/> and <see cref="Upper"/>.</summary>
    public double Confidence { get; }

    /// <summary>The normal critical value used for the intervals, <c>1.96</c> at 95%.</summary>
    public double CriticalValue { get; }

    /// <summary>The innovation variance the intervals are built from.</summary>
    public double InnovationVariance { get; }

    /// <summary>Point forecasts for steps <c>1 .. Steps</c>, on the original scale.</summary>
    public ReadOnlyMemory<double> Mean => _mean;

    /// <summary>Lower interval bounds, aligned with <see cref="Mean"/>.</summary>
    public ReadOnlyMemory<double> Lower => _lower;

    /// <summary>Upper interval bounds, aligned with <see cref="Mean"/>.</summary>
    public ReadOnlyMemory<double> Upper => _upper;

    /// <summary>Forecast-error standard deviations per step, on the original scale.</summary>
    public ReadOnlyMemory<double> StandardError => _standardError;

    /// <summary>
    /// The ψ-weights on the original scale, <c>psi_0 .. psi_(Steps-1)</c>: the forecast
    /// error at step <c>h</c> is <c>sum_(j&lt;h) psi_j e_(origin+h-j)</c>. Exposed so any
    /// linear function of the path can be given a correct variance.
    /// </summary>
    public ReadOnlyMemory<double> PsiWeights => _psi;

    /// <summary>
    /// Anything the caller should know before trusting the numbers — a non-invertible
    /// moving-average part, for instance. Empty when there is nothing to say.
    /// </summary>
    public IReadOnlyList<string> Warnings => _warnings;

    /// <summary>
    /// The sum of the forecast over steps <c>[start, start + count)</c> (1-based steps),
    /// with a standard error that includes the covariance between steps.
    /// </summary>
    /// <param name="start">The first step to include, from 1.</param>
    /// <param name="count">How many steps to include.</param>
    /// <returns>The aggregate and its interval at this result's confidence level.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The range does not lie inside <c>1 .. Steps</c>.</exception>
    public ForecastAggregate Sum(int start, int count)
    {
        if (start < 1 || start > Steps)
        {
            throw new ArgumentOutOfRangeException(nameof(start), start, $"Start must lie in 1..{Steps}.");
        }

        if (count < 1 || start + count - 1 > Steps)
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, $"Count must lie in 1..{Steps - start + 1}.");
        }

        var last = start + count - 1;
        var mean = 0d;

        for (var h = start; h <= last; h++)
        {
            mean += _mean[h - 1];
        }

        // Error of the sum = sum_h sum_(j<h) psi_j e_(h-j). Grouping by innovation k = h - j:
        // the coefficient on e_k is sum over the included steps h >= k of psi_(h-k).
        var variance = 0d;

        for (var k = 1; k <= last; k++)
        {
            var coefficient = 0d;

            for (var h = Math.Max(start, k); h <= last; h++)
            {
                coefficient += _psi[h - k];
            }

            variance += coefficient * coefficient;
        }

        var standardError = Math.Sqrt(InnovationVariance * variance);
        return new ForecastAggregate(start, count, mean, standardError, CriticalValue);
    }
}

/// <summary>A linear aggregate of a forecast path with its own interval.</summary>
public readonly record struct ForecastAggregate
{
    internal ForecastAggregate(int start, int count, double mean, double standardError, double criticalValue)
    {
        Start = start;
        Count = count;
        Mean = mean;
        StandardError = standardError;
        Lower = mean - (criticalValue * standardError);
        Upper = mean + (criticalValue * standardError);
    }

    /// <summary>The first step included, from 1.</summary>
    public int Start { get; }

    /// <summary>The number of steps included.</summary>
    public int Count { get; }

    /// <summary>The aggregate point forecast.</summary>
    public double Mean { get; }

    /// <summary>The standard error of the aggregate, covariance included.</summary>
    public double StandardError { get; }

    /// <summary>The lower interval bound.</summary>
    public double Lower { get; }

    /// <summary>The upper interval bound.</summary>
    public double Upper { get; }
}
