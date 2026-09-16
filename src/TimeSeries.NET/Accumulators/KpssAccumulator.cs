namespace TimeSeries.Accumulators;

/// <summary>
/// Accumulates the KPSS test for level stationarity in one pass: the partial-sum
/// statistic through three running sums, and the long-run variance through the
/// classical autocovariances it shares a lag window with.
/// </summary>
/// <remarks>
/// <para>
/// With <c>e_t = y_t - mean</c> and <c>S_t = sum_(i&lt;=t) e_i</c>, the statistic is
/// <c>eta = sum S_t^2 / (n^2 s^2_LR)</c>. The mean is unknown until the pass ends, but
/// <c>S_t = C_t - t * mean</c> where <c>C_t = sum_(i&lt;=t) y_i</c>, so
/// <c>sum S_t^2 = sum C_t^2 - 2 mean sum t C_t + mean^2 sum t^2</c> — three sums that need
/// no mean at all. The caller shifts values by an offset near the data first, which keeps
/// the cancellation in that expansion harmless.
/// </para>
/// <para>
/// Information criteria cannot choose a differencing order: models at different
/// <c>d</c> are fitted to different series. This test can. Under the null the series is
/// stationary about its mean; a statistic above the critical value rejects that and
/// calls for another difference.
/// </para>
/// </remarks>
public sealed class KpssAccumulator : IAccumulator<KpssAccumulator, KpssResult>
{
    private readonly AutocovarianceAccumulator _autocovariance;
    private double _sum;
    private double _sumCompensation;
    private double _cumulative;
    private double _sumCumulativeSquared;
    private double _sumCumulativeSquaredCompensation;
    private double _sumIndexCumulative;
    private double _sumIndexCumulativeCompensation;
    private double _sumIndexSquared;
    private double _count;

    /// <summary>
    /// Creates an accumulator.
    /// </summary>
    /// <param name="maxLag">
    /// The deepest autocovariance kept for the long-run variance. The bandwidth actually
    /// used is <c>min(maxLag, floor(4 (n/100)^(1/4)))</c>, Kwiatkowski et al.'s <c>l4</c>.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is negative.</exception>
    public KpssAccumulator(int maxLag = 32)
    {
        _autocovariance = new AutocovarianceAccumulator(maxLag);
    }

    /// <summary>The deepest autocovariance kept.</summary>
    public int MaxLag => _autocovariance.MaxLag;

    /// <inheritdoc />
    public double Count => _count;

    /// <summary>
    /// Folds one observation in.
    /// </summary>
    /// <param name="row">
    /// The lag row ending at the current observation; <c>row[0]</c> is the value, and the
    /// lags feed the long-run variance. Values should already be shifted by an offset near
    /// the data.
    /// </param>
    public void Add(ReadOnlySpan<double> row)
    {
        if (row.Length == 0)
        {
            throw new ArgumentException("A lag row must hold at least the current observation.", nameof(row));
        }

        _autocovariance.Add(row.Length > MaxLag + 1 ? row.Slice(0, MaxLag + 1) : row);

        var value = row[0];
        _count++;
        Compensated.Add(ref _sum, ref _sumCompensation, value);
        _cumulative += value;

        Compensated.Add(ref _sumCumulativeSquared, ref _sumCumulativeSquaredCompensation, _cumulative * _cumulative);
        Compensated.Add(ref _sumIndexCumulative, ref _sumIndexCumulativeCompensation, _count * _cumulative);
        _sumIndexSquared += _count * _count;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Not supported: the partial sums are position-dependent, so two ranges cannot be
    /// added without knowing where one ends and the other begins. The scan that owns this
    /// accumulator is sequential.
    /// </remarks>
    /// <exception cref="NotSupportedException">Always.</exception>
    public void Merge(in KpssAccumulator other) =>
        throw new NotSupportedException("KPSS partial sums are position-dependent and cannot be merged across partitions.");

    /// <inheritdoc />
    /// <remarks>Not meaningful for a test statistic; the call is ignored.</remarks>
    public void Scale(double lambda)
    {
    }

    /// <inheritdoc />
    public KpssResult Freeze()
    {
        var n = _count;
        var autocovariances = _autocovariance.Freeze();

        if (n < 4d)
        {
            return new KpssResult(double.NaN, 0, n, double.NaN, autocovariances);
        }

        var mean = (_sum + _sumCompensation) / n;
        var sumSquaredPartialSums = (_sumCumulativeSquared + _sumCumulativeSquaredCompensation)
            - (2d * mean * (_sumIndexCumulative + _sumIndexCumulativeCompensation))
            + (mean * mean * _sumIndexSquared);

        var gamma = autocovariances.Gamma.Span;
        var bandwidth = Math.Min(MaxLag, (int)Math.Floor(4d * Math.Pow(n / 100d, 0.25)));
        var longRunVariance = gamma[0];

        for (var k = 1; k <= bandwidth; k++)
        {
            longRunVariance += 2d * (1d - (k / (bandwidth + 1d))) * gamma[k];
        }

        if (!(longRunVariance > 0d))
        {
            // A constant series: trivially stationary.
            return new KpssResult(0d, bandwidth, n, 0d, autocovariances);
        }

        return new KpssResult(Math.Max(sumSquaredPartialSums, 0d) / (n * n * longRunVariance), bandwidth, n, longRunVariance, autocovariances);
    }

    /// <summary>Discards all accumulated state.</summary>
    public void Reset()
    {
        _autocovariance.Reset();
        _sum = 0d;
        _sumCompensation = 0d;
        _cumulative = 0d;
        _sumCumulativeSquared = 0d;
        _sumCumulativeSquaredCompensation = 0d;
        _sumIndexCumulative = 0d;
        _sumIndexCumulativeCompensation = 0d;
        _sumIndexSquared = 0d;
        _count = 0d;
    }
}

/// <summary>The outcome of a KPSS level-stationarity test.</summary>
public sealed class KpssResult
{
    internal KpssResult(double statistic, int bandwidth, double count, double longRunVariance, Autocovariances autocovariances)
    {
        Statistic = statistic;
        Bandwidth = bandwidth;
        Count = count;
        LongRunVariance = longRunVariance;
        Autocovariances = autocovariances;
    }

    /// <summary>The autocovariances of the tested series, to the accumulator's depth.</summary>
    public Autocovariances Autocovariances { get; }

    /// <summary>The test statistic <c>eta</c>. NaN when fewer than four observations were seen.</summary>
    public double Statistic { get; }

    /// <summary>The Bartlett bandwidth used for the long-run variance.</summary>
    public int Bandwidth { get; }

    /// <summary>Observations the statistic is built from.</summary>
    public double Count { get; }

    /// <summary>The Bartlett-kernel long-run variance estimate.</summary>
    public double LongRunVariance { get; }

    /// <summary>
    /// The asymptotic critical value for the level test at a significance level:
    /// 0.347 (10%), 0.463 (5%), 0.574 (2.5%), 0.739 (1%).
    /// </summary>
    /// <param name="significance">One of 0.10, 0.05, 0.025 or 0.01.</param>
    /// <returns>The critical value.</returns>
    /// <exception cref="ArgumentOutOfRangeException">An unsupported significance level.</exception>
    public static double CriticalValue(double significance) => significance switch
    {
        0.10 => 0.347,
        0.05 => 0.463,
        0.025 => 0.574,
        0.01 => 0.739,
        _ => throw new ArgumentOutOfRangeException(nameof(significance), significance, "Supported levels are 0.10, 0.05, 0.025 and 0.01."),
    };

    /// <summary>
    /// True when the statistic exceeds the critical value, rejecting stationarity — the
    /// series wants another difference.
    /// </summary>
    /// <param name="significance">The significance level; default 5%.</param>
    /// <returns>Whether stationarity is rejected. False when the statistic is NaN.</returns>
    public bool RejectsStationarity(double significance = 0.05) => Statistic > CriticalValue(significance);
}
