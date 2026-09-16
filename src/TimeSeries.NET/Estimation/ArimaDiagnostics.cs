using TimeSeries.Solvers;

namespace TimeSeries;

/// <summary>What the estimator can say about a fit beyond its coefficients.</summary>
public sealed class ArimaDiagnostics
{
    internal ArimaDiagnostics(
        double effectiveObservations,
        int parameterCount,
        double residualSumOfSquares,
        double logLikelihood,
        double aic,
        double aicc,
        double bic,
        int pilotOrder,
        bool isStationary,
        double stationarityMargin,
        bool isInvertible,
        double invertibilityMargin,
        bool isConstantSeries,
        SolveDiagnostics solve)
    {
        EffectiveObservations = effectiveObservations;
        ParameterCount = parameterCount;
        ResidualSumOfSquares = residualSumOfSquares;
        LogLikelihood = logLikelihood;
        Aic = aic;
        Aicc = aicc;
        Bic = bic;
        PilotOrder = pilotOrder;
        IsStationary = isStationary;
        StationarityMargin = stationarityMargin;
        IsInvertible = isInvertible;
        InvertibilityMargin = invertibilityMargin;
        IsConstantSeries = isConstantSeries;
        Solve = solve;
    }

    /// <summary>
    /// The number of rows that entered the normal equations: differenced observations
    /// less the lag depth, after any forgetting factor.
    /// </summary>
    public double EffectiveObservations { get; }

    /// <summary>Coefficients estimated, plus one for the innovation variance.</summary>
    public int ParameterCount { get; }

    /// <summary>The residual sum of squares of the second-stage regression.</summary>
    public double ResidualSumOfSquares { get; }

    /// <summary>The Gaussian conditional log-likelihood at the estimates.</summary>
    public double LogLikelihood { get; }

    /// <summary>Akaike's information criterion.</summary>
    public double Aic { get; }

    /// <summary>AIC with the small-sample correction; the criterion the library selects by.</summary>
    public double Aicc { get; }

    /// <summary>The Bayesian information criterion.</summary>
    public double Bic { get; }

    /// <summary>The pilot autoregression order used, or 0 when the model has no MA part.</summary>
    public int PilotOrder { get; }

    /// <summary>True when every root of the AR polynomial lies outside the unit circle.</summary>
    public bool IsStationary { get; }

    /// <summary>How far inside the stationary region the AR polynomial sits; 1 when p = 0.</summary>
    public double StationarityMargin { get; }

    /// <summary>
    /// True when every root of the MA polynomial lies outside the unit circle. A
    /// non-invertible fit forecasts, but its intervals should not be trusted.
    /// </summary>
    public bool IsInvertible { get; }

    /// <summary>How far inside the invertible region the MA polynomial sits; 1 when q = 0.</summary>
    public double InvertibilityMargin { get; }

    /// <summary>
    /// True when the differenced series was exactly constant and the fit was
    /// short-circuited to the deterministic model: intercept equal to that constant,
    /// every other coefficient zero, innovation variance zero.
    /// </summary>
    public bool IsConstantSeries { get; }

    /// <summary>How the second-stage normal equations solved.</summary>
    public SolveDiagnostics Solve { get; }
}
