using TimeSeries.Transforms;

namespace TimeSeries;

/// <summary>
/// The bounded tail a fit carries so it can forecast: how to undo the differencing, how
/// to keep differencing the regressors, the last <c>p</c> differenced values, and the
/// last <c>q</c> innovations. Sized by the model order, never by the series.
/// </summary>
internal sealed class ForecastSeed
{
    internal ForecastSeed(
        IntegrationState integration,
        IntegrationState[] regressorStates,
        double[] recentValues,
        double[] recentResiduals)
    {
        Integration = integration;
        RegressorStates = regressorStates;
        RecentValues = recentValues;
        RecentResiduals = recentResiduals;
    }

    /// <summary>Inverts the differencing from the end of the fitted series onward.</summary>
    internal IntegrationState Integration { get; }

    /// <summary>
    /// One per regressor: the differencing state at the end of the fitted regressors, so
    /// future raw regressor values can be differenced the same way the fitted ones were.
    /// </summary>
    internal IntegrationState[] RegressorStates { get; }

    /// <summary>The last <c>p</c> values on the differenced scale, oldest first.</summary>
    internal double[] RecentValues { get; }

    /// <summary>The last <c>q</c> innovations, oldest first.</summary>
    internal double[] RecentResiduals { get; }
}
