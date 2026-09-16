namespace TimeSeries;

/// <summary>
/// A fitted ARIMA or ARIMAX model. Immutable, thread-safe, and free of any reference to
/// the series it came from: a few dozen doubles plus a tail bounded by the model order.
/// </summary>
/// <remarks>
/// On the differenced scale the fitted model is
/// <code>
/// z_t = Intercept + sum_k AutoRegressive[k-1] z_(t-k)
///                 + e_t + sum_j MovingAverage[j-1] e_(t-j)
///                 + sum_i ExogenousCoefficients[i] x_(t,i)
/// </code>
/// with <c>e_t</c> white noise of variance <see cref="InnovationVariance"/>.
/// </remarks>
public sealed partial class ArimaFit
{
    private readonly double[] _phi;
    private readonly double[] _theta;
    private readonly double[] _beta;
    private readonly double[] _standardErrors;

    internal ArimaFit(
        ArimaOptions options,
        double[] phi,
        double[] theta,
        double[] beta,
        double intercept,
        double innovationVariance,
        long observationCount,
        ArimaDiagnostics diagnostics,
        FitWindow window,
        double[] standardErrors,
        ForecastSeed seed)
    {
        Options = options;
        _phi = phi;
        _theta = theta;
        _beta = beta;
        Intercept = intercept;
        InnovationVariance = innovationVariance;
        ObservationCount = observationCount;
        Diagnostics = diagnostics;
        Window = window;
        _standardErrors = standardErrors;
        Seed = seed;
    }

    /// <summary>The options the model was fitted with.</summary>
    public ArimaOptions Options { get; }

    /// <summary>The non-seasonal order <c>(p, d, q)</c>.</summary>
    public ArimaOrder Order => Options.Order;

    /// <summary>The seasonal differencing applied.</summary>
    public SeasonalOrder Seasonal => Options.Seasonal;

    /// <summary>Autoregressive coefficients <c>phi_1 .. phi_p</c>.</summary>
    public ReadOnlyMemory<double> AutoRegressive => _phi;

    /// <summary>Moving-average coefficients <c>theta_1 .. theta_q</c>, additive convention.</summary>
    public ReadOnlyMemory<double> MovingAverage => _theta;

    /// <summary>Exogenous coefficients <c>beta_1 .. beta_r</c>, in the regressors' column order.</summary>
    public ReadOnlyMemory<double> ExogenousCoefficients => _beta;

    /// <summary>The number of exogenous regressors the model was fitted with.</summary>
    public int RegressorCount => _beta.Length;

    /// <summary>
    /// The constant on the differenced scale, or zero when
    /// <see cref="ArimaOptions.IncludeIntercept"/> was false. With <c>d = 1</c> this is the
    /// drift per period.
    /// </summary>
    public double Intercept { get; }

    /// <summary>The innovation variance <c>sigma-hat^2</c>, residual sum of squares over effective observations.</summary>
    public double InnovationVariance { get; }

    /// <summary>Raw observations the model was fitted over, before differencing.</summary>
    public long ObservationCount { get; }

    /// <summary>Information criteria, stability checks and solve health.</summary>
    public ArimaDiagnostics Diagnostics { get; }

    /// <summary>The row range fitted, for audit and as the incremental watermark.</summary>
    public FitWindow Window { get; }

    /// <summary>
    /// Approximate standard errors from the second-stage regression, ordered as the
    /// coefficients: intercept (when included), then autoregressive, moving-average and
    /// exogenous. They treat the pilot residuals as observed, so they are slightly
    /// optimistic on short series; NaN for a constant series.
    /// </summary>
    public ReadOnlyMemory<double> StandardErrors => _standardErrors;

    /// <summary>What the forecaster needs from the end of the fitted series.</summary>
    internal ForecastSeed Seed { get; }

    /// <summary>
    /// Forecasts an ARIMA model over a horizon.
    /// </summary>
    /// <param name="horizon">How far ahead, in periods or calendar time.</param>
    /// <param name="options">Interval confidence and invertibility policy; null for the defaults.</param>
    /// <returns>The forecast path with prediction intervals.</returns>
    /// <exception cref="ForecastHorizonException">A calendar horizon cannot be resolved; see <see cref="ForecastHorizon.Resolve"/>.</exception>
    /// <exception cref="ArgumentException">The model has exogenous regressors, whose future values are required.</exception>
    /// <exception cref="InvalidOperationException">The model is not invertible and <see cref="ForecastOptions.RequireInvertible"/> is set.</exception>
    public ForecastResult Forecast(ForecastHorizon horizon, ForecastOptions? options = null)
        => Forecaster.Forecast(this, horizon.Resolve(Options.Frequency), null, options ?? ForecastOptions.Default);

    /// <summary>
    /// Forecasts an ARIMAX model over a horizon, given the regressors' future values.
    /// </summary>
    /// <param name="horizon">How far ahead, in periods or calendar time.</param>
    /// <param name="future">
    /// One row of regressors per step of the horizon, on the raw (undifferenced) scale.
    /// The intervals are conditional on these values and carry none of their uncertainty.
    /// </param>
    /// <param name="options">Interval confidence and invertibility policy; null for the defaults.</param>
    /// <returns>The forecast path with prediction intervals.</returns>
    /// <exception cref="ForecastHorizonException">A calendar horizon cannot be resolved; see <see cref="ForecastHorizon.Resolve"/>.</exception>
    /// <exception cref="ArgumentNullException"><paramref name="future"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="future"/> has the wrong number of rows or regressors for this model and horizon.
    /// </exception>
    /// <exception cref="InvalidSeriesException">A future regressor value is NaN or infinite.</exception>
    /// <exception cref="InvalidOperationException">The model is not invertible and <see cref="ForecastOptions.RequireInvertible"/> is set.</exception>
    public ForecastResult Forecast(ForecastHorizon horizon, ExogenousMatrix future, ForecastOptions? options = null)
    {
        if (future is null)
        {
            throw new ArgumentNullException(nameof(future));
        }

        return Forecaster.Forecast(this, horizon.Resolve(Options.Frequency), future, options ?? ForecastOptions.Default);
    }
}
