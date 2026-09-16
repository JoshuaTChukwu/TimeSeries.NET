namespace TimeSeries;

/// <summary>
/// A calendar horizon could not be resolved to a number of periods — because the model
/// declares no <see cref="SeriesFrequency"/>, or because it resolves to fewer than one
/// period. The library refuses to guess rather than silently forecast the wrong span.
/// </summary>
public sealed class ForecastHorizonException : InvalidOperationException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="message">What was asked and why it could not be resolved.</param>
    public ForecastHorizonException(string message)
        : base(message)
    {
    }
}
