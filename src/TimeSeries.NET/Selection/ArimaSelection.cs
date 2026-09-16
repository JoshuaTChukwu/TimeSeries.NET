using TimeSeries.Accumulators;

namespace TimeSeries;

/// <summary>One model tried by <see cref="AutoArima"/>, with what happened.</summary>
public sealed class ArimaCandidate
{
    internal ArimaCandidate(ArimaOrder order, SeasonalOrder seasonal, ArimaFit? fit, Exception? failure)
    {
        Order = order;
        Seasonal = seasonal;
        Fit = fit;
        Failure = failure;
    }

    /// <summary>The order tried.</summary>
    public ArimaOrder Order { get; }

    /// <summary>The seasonal differencing tried.</summary>
    public SeasonalOrder Seasonal { get; }

    /// <summary>The fitted model, or null when the fit failed.</summary>
    public ArimaFit? Fit { get; }

    /// <summary>Why the fit failed, or null on success.</summary>
    public Exception? Failure { get; }

    /// <summary>True when <see cref="Fit"/> is present.</summary>
    public bool Succeeded => Fit is not null;

    /// <summary>The candidate's AIC, or NaN when it failed.</summary>
    public double Aic => Fit?.Diagnostics.Aic ?? double.NaN;

    /// <summary>The candidate's AICc, or NaN when it failed.</summary>
    public double Aicc => Fit?.Diagnostics.Aicc ?? double.NaN;

    /// <summary>The candidate's BIC, or NaN when it failed.</summary>
    public double Bic => Fit?.Diagnostics.Bic ?? double.NaN;

    internal double Score(InformationCriterion criterion) => criterion switch
    {
        InformationCriterion.Aic => Aic,
        InformationCriterion.Bic => Bic,
        _ => Aicc,
    };
}

/// <summary>How one differencing candidate fared in the stationarity test.</summary>
public sealed class DifferencingCandidate
{
    internal DifferencingCandidate(int order, SeasonalOrder seasonal, KpssResult kpss, double seasonalAutocorrelation, bool isConstant)
    {
        Order = order;
        Seasonal = seasonal;
        Kpss = kpss;
        SeasonalAutocorrelation = seasonalAutocorrelation;
        IsConstant = isConstant;
    }

    /// <summary>The non-seasonal order <c>d</c>.</summary>
    public int Order { get; }

    /// <summary>The seasonal differencing.</summary>
    public SeasonalOrder Seasonal { get; }

    /// <summary>The KPSS result on the differenced series.</summary>
    public KpssResult Kpss { get; }

    /// <summary>
    /// The autocorrelation of the differenced series at the seasonal period, or NaN when
    /// no seasonal period is in play.
    /// </summary>
    public double SeasonalAutocorrelation { get; }

    /// <summary>True when the differenced series is exactly constant.</summary>
    public bool IsConstant { get; }
}

/// <summary>
/// The outcome of an automatic order search: the winner, and the whole ranked table so
/// the choice is inspectable rather than opaque.
/// </summary>
public sealed class ArimaSelection
{
    internal ArimaSelection(
        ArimaFit best,
        IReadOnlyList<ArimaCandidate> candidates,
        IReadOnlyList<DifferencingCandidate> differencing,
        DifferencingCandidate chosenDifferencing,
        InformationCriterion criterion)
    {
        Best = best;
        Candidates = candidates;
        Differencing = differencing;
        ChosenDifferencing = chosenDifferencing;
        Criterion = criterion;
    }

    /// <summary>The winning fit.</summary>
    public ArimaFit Best { get; }

    /// <summary>Every <c>(p, q)</c> tried at the chosen differencing, best first; failures last.</summary>
    public IReadOnlyList<ArimaCandidate> Candidates { get; }

    /// <summary>Every differencing tested, in the order tested.</summary>
    public IReadOnlyList<DifferencingCandidate> Differencing { get; }

    /// <summary>The differencing the grid was run at, and why.</summary>
    public DifferencingCandidate ChosenDifferencing { get; }

    /// <summary>The criterion that ranked <see cref="Candidates"/>.</summary>
    public InformationCriterion Criterion { get; }
}
