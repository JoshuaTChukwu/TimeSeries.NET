namespace TimeSeries;

/// <summary>
/// Rows arrived out of order — a key that decreased or reappeared during a grouped scan,
/// or a timestamp that did not advance. Fitting would silently mix series or fold time
/// back on itself, so the scan stops here.
/// </summary>
public sealed class SeriesOrderException : InvalidOperationException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="previous">The key or timestamp seen before, as text.</param>
    /// <param name="current">The key or timestamp that violated the order, as text.</param>
    /// <param name="message">What was expected and what arrived.</param>
    public SeriesOrderException(string previous, string current, string message)
        : base(message)
    {
        Previous = previous;
        Current = current;
    }

    /// <summary>The key or timestamp seen before.</summary>
    public string Previous { get; }

    /// <summary>The key or timestamp that violated the order.</summary>
    public string Current { get; }
}
