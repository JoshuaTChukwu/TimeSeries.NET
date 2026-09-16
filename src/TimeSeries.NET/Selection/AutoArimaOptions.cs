namespace TimeSeries;

/// <summary>Which information criterion ranks candidate orders.</summary>
public enum InformationCriterion
{
    /// <summary>Akaike's information criterion.</summary>
    Aic,

    /// <summary>AIC with the small-sample correction. The default.</summary>
    Aicc,

    /// <summary>The Bayesian information criterion; penalises order more heavily.</summary>
    Bic,
}

/// <summary>
/// The search space for <see cref="AutoArima"/>.
/// </summary>
/// <remarks>
/// <para>
/// Differencing order is <em>not</em> chosen by information criterion — models at
/// different <c>d</c> are fitted to different series, so their likelihoods are not
/// comparable. It is chosen by a KPSS test on each candidate: the smallest <c>d</c> whose
/// differenced series is not rejected as non-stationary, or the largest candidate if all
/// are. Then <c>(p, q)</c> are ranked by <see cref="Criterion"/> at that fixed <c>d</c>,
/// where the comparison is valid.
/// </para>
/// <para>
/// Every candidate is a different small projection of the same accumulated matrix, so the
/// whole grid — every <c>d</c>, every <c>(p, q)</c> — costs one pass over the data.
/// </para>
/// </remarks>
public sealed record AutoArimaOptions
{
    /// <summary>The largest autoregressive order tried. Default 5.</summary>
    public int MaxP { get; init; } = 5;

    /// <summary>The largest moving-average order tried. Default 5.</summary>
    public int MaxQ { get; init; } = 5;

    /// <summary>Candidate differencing orders, tested in ascending order. Default 0, 1, 2.</summary>
    public IReadOnlyList<int> DifferenceOrders { get; init; } = [0, 1, 2];

    /// <summary>
    /// Candidate seasonal differencings. Default none. With more than one candidate, the
    /// smallest whose <c>d</c>-differenced series has a lag-<c>s</c> autocorrelation below
    /// <see cref="SeasonalStrengthThreshold"/> is chosen — a heuristic, stated as such;
    /// a seasonal unit-root test is a later addition.
    /// </summary>
    public IReadOnlyList<SeasonalOrder> SeasonalCandidates { get; init; } = [SeasonalOrder.None];

    /// <summary>Lag-<c>s</c> autocorrelation above which another seasonal difference is taken. Default 0.5.</summary>
    public double SeasonalStrengthThreshold { get; init; } = 0.5;

    /// <summary>The criterion that ranks <c>(p, q)</c>. Default <see cref="InformationCriterion.Aicc"/>.</summary>
    public InformationCriterion Criterion { get; init; } = InformationCriterion.Aicc;

    /// <summary>Significance level of the KPSS test that chooses <c>d</c>. Default 0.05.</summary>
    public double StationaritySignificance { get; init; } = 0.05;

    /// <summary>Whether candidate models include a constant. Default true.</summary>
    public bool IncludeIntercept { get; init; } = true;

    /// <summary>The largest pilot order for the Hannan-Rissanen second stage. Default 32.</summary>
    public int MaxPilotOrder { get; init; } = 32;

    /// <summary>A ridge for the normal equations; zero disables. Default 0.</summary>
    public double Ridge { get; init; }

    /// <summary>The series' frequency, carried into the chosen model for calendar horizons.</summary>
    public SeriesFrequency? Frequency { get; init; }

    /// <summary>Residual autocorrelations examined by the Ljung-Box test on every candidate. Default 10; zero disables.</summary>
    public int LjungBoxLags { get; init; } = 10;

    /// <summary>
    /// Whether a candidate must be stationary and invertible to win. Default true, as in
    /// auto.arima. Hannan-Rissanen on a short series can land an over-parameterised
    /// candidate outside the admissible region; its in-sample criterion can still look
    /// best while its forecasts diverge. With this on, admissible candidates rank ahead
    /// of inadmissible ones whatever their criterion; the ranked table still lists all.
    /// </summary>
    public bool RequireAdmissible { get; init; } = true;

    /// <summary>Checks the options.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A bound is negative, a list is empty, or the significance level is unsupported.</exception>
    public void Validate()
    {
        if (MaxP < 0 || MaxQ < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxP), $"MaxP ({MaxP}) and MaxQ ({MaxQ}) must be zero or greater.");
        }

        if (DifferenceOrders is null || DifferenceOrders.Count == 0 || DifferenceOrders.Any(d => d < 0))
        {
            throw new ArgumentOutOfRangeException(nameof(DifferenceOrders), "At least one non-negative differencing order is required.");
        }

        if (SeasonalCandidates is null || SeasonalCandidates.Count == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(SeasonalCandidates), "At least one seasonal candidate is required (SeasonalOrder.None for none).");
        }

        if (MaxPilotOrder < Math.Max(1, MaxP + MaxQ) && MaxQ > 0)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxPilotOrder), MaxPilotOrder, $"MaxPilotOrder must be at least MaxP + MaxQ = {MaxP + MaxQ}.");
        }

        _ = Accumulators.KpssResult.CriticalValue(StationaritySignificance);

        Widest(DifferenceOrders[0], SeasonalCandidates[0]).Validate();
    }

    /// <summary>The scan-defining model for one differencing: the largest orders, so every smaller candidate is a sub-problem.</summary>
    internal ArimaOptions Widest(int d, SeasonalOrder seasonal) => new()
    {
        Order = new(MaxP, d, MaxQ),
        Seasonal = seasonal,
        IncludeIntercept = IncludeIntercept,
        MaxPilotOrder = MaxPilotOrder,
        Ridge = Ridge,
        Frequency = Frequency,
        LjungBoxLags = LjungBoxLags,
    };

    /// <summary>One candidate model.</summary>
    internal ArimaOptions Candidate(int p, int d, int q, SeasonalOrder seasonal) => new()
    {
        Order = new(p, d, q),
        Seasonal = seasonal,
        IncludeIntercept = IncludeIntercept,
        MaxPilotOrder = MaxPilotOrder,
        Ridge = Ridge,
        Frequency = Frequency,
        LjungBoxLags = LjungBoxLags,
    };
}
