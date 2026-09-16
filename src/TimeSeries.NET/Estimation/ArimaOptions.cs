namespace TimeSeries;

/// <summary>
/// Everything that describes an ARIMA or ARIMAX model before it is fitted. Immutable;
/// share freely.
/// </summary>
/// <remarks>
/// The model, on the differenced scale <c>z_t = (1-B)^d (1-B^s)^D y_t</c>, is
/// <code>
/// z_t = c + phi_1 z_(t-1) + ... + phi_p z_(t-p)
///         + e_t + theta_1 e_(t-1) + ... + theta_q e_(t-q)
///         + beta' x_t
/// </code>
/// with the moving-average terms in the additive sign convention (as in statsmodels), and
/// the exogenous regressors differenced by the same operator as the series.
/// </remarks>
public sealed record ArimaOptions
{
    /// <summary>The non-seasonal order <c>(p, d, q)</c>. Default <c>(1, 1, 1)</c>.</summary>
    public ArimaOrder Order { get; init; } = new(1, 1, 1);

    /// <summary>Seasonal differencing. Default none.</summary>
    public SeasonalOrder Seasonal { get; init; } = SeasonalOrder.None;

    /// <summary>
    /// Whether to estimate a constant <c>c</c> on the differenced scale. With <c>d = 1</c>
    /// this is the drift. Default true.
    /// </summary>
    public bool IncludeIntercept { get; init; } = true;

    /// <summary>
    /// The order <c>m</c> of the long autoregression whose residuals stand in for the
    /// innovations in the Hannan-Rissanen second stage. Null selects it by AICc among
    /// <c>p + q .. MaxPilotOrder</c>, at no data cost. Ignored when <c>q = 0</c>.
    /// </summary>
    public int? PilotOrder { get; init; }

    /// <summary>The largest pilot order considered when <see cref="PilotOrder"/> is null. Default 32.</summary>
    public int MaxPilotOrder { get; init; } = 32;

    /// <summary>
    /// A ridge added to the scaled normal equations. Zero, the default, disables it; a
    /// small value such as 1e-6 rescues a nearly collinear design at the cost of slight bias.
    /// </summary>
    public double Ridge { get; init; }

    /// <summary>
    /// Exponential forgetting factor <c>lambda</c> in (0, 1], applied by the incremental
    /// estimator before each fold. 1, the default, forgets nothing. Has no effect on a
    /// single fit.
    /// </summary>
    public double ForgettingFactor { get; init; } = 1d;

    /// <summary>
    /// The series' observation frequency. Required only to resolve calendar horizons such
    /// as <c>ForecastHorizon.Years(2)</c>; a horizon in periods never needs it.
    /// </summary>
    public SeriesFrequency? Frequency { get; init; }

    /// <summary>
    /// The number of residual autocorrelations <c>K</c> the Ljung-Box test examines.
    /// Default 10; zero disables the test. Computed exactly in the same pass by carrying
    /// the Gram matrix <c>K</c> lags deeper, which costs <c>K</c> extra lag columns and
    /// the first <c>K</c> rows of the differenced series — both stages and the test then
    /// share one row range.
    /// </summary>
    public int LjungBoxLags { get; init; } = 10;

    /// <summary>
    /// Checks the options for consistency.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// An order is negative; a seasonal difference has a period below 2; the forgetting
    /// factor is outside (0, 1]; the ridge is negative; or a pilot order is too small for
    /// the model or larger than <see cref="MaxPilotOrder"/>.
    /// </exception>
    public void Validate()
    {
        if (Order.P < 0 || Order.D < 0 || Order.Q < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(Order), Order, "Every order must be zero or greater.");
        }

        if (Seasonal.D < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(Seasonal), Seasonal, "Seasonal order must be zero or greater.");
        }

        if (Seasonal.D > 0 && Seasonal.Period < 2)
        {
            throw new ArgumentOutOfRangeException(
                nameof(Seasonal), Seasonal, "A seasonal difference needs a period of 2 or greater.");
        }

        if (!(ForgettingFactor > 0d && ForgettingFactor <= 1d))
        {
            throw new ArgumentOutOfRangeException(
                nameof(ForgettingFactor), ForgettingFactor, "The forgetting factor must lie in (0, 1].");
        }

        if (!(Ridge >= 0d))
        {
            throw new ArgumentOutOfRangeException(nameof(Ridge), Ridge, "Ridge must be zero or greater.");
        }

        if (LjungBoxLags < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(LjungBoxLags), LjungBoxLags, "Ljung-Box lags must be zero or greater.");
        }

        if (Order.Q == 0)
        {
            return;
        }

        var minimumPilot = Math.Max(1, Order.P + Order.Q);

        if (PilotOrder is int pilot)
        {
            if (pilot < minimumPilot)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(PilotOrder), pilot,
                    $"The pilot autoregression must have order at least p + q = {minimumPilot} " +
                    "for its residuals to stand in for the innovations.");
            }
        }
        else if (MaxPilotOrder < minimumPilot)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MaxPilotOrder), MaxPilotOrder,
                $"MaxPilotOrder must be at least p + q = {minimumPilot}.");
        }
    }

    /// <summary>The combined differencing operator this model applies.</summary>
    internal Transforms.DifferenceSpec Differencing => new(Order.D, Seasonal.D, Seasonal.Period);

    /// <summary>
    /// The pilot orders considered: the fixed one, or <c>p + q .. MaxPilotOrder</c>.
    /// Empty when the model has no moving-average part.
    /// </summary>
    internal (int Minimum, int Maximum) PilotRange
    {
        get
        {
            if (Order.Q == 0)
            {
                return (0, 0);
            }

            if (PilotOrder is int pilot)
            {
                return (pilot, pilot);
            }

            return (Math.Max(1, Order.P + Order.Q), MaxPilotOrder);
        }
    }

    /// <summary>The lag depth <c>L = max(p, q + m_max)</c> estimation needs.</summary>
    internal int LagDepth => Order.Q == 0 ? Order.P : Math.Max(Order.P, Order.Q + PilotRange.Maximum);

    /// <summary>The lag depth the Gram matrix actually spans: <c>L</c> plus the Ljung-Box lags.</summary>
    internal int GramDepth => LagDepth + LjungBoxLags;

    /// <summary>The deepest exogenous lag kept in the row: <c>q</c> plus the Ljung-Box lags.</summary>
    internal int ExogenousLagDepth => Order.Q + LjungBoxLags;
}
