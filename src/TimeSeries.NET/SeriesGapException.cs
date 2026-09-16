namespace TimeSeries;

/// <summary>
/// Two consecutive observations are further apart than the declared step, and the gap
/// policy is to refuse. ARIMA assumes regular spacing; a silently misaligned fit is the
/// worst outcome, so the default is to throw and name the timestamps either side.
/// </summary>
public sealed class SeriesGapException : InvalidOperationException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="before">The observation before the gap, as text.</param>
    /// <param name="after">The observation after the gap, as text.</param>
    /// <param name="key">The series key when scanning a population, or null.</param>
    /// <param name="message">The gap, the expected step, and what to do.</param>
    public SeriesGapException(string before, string after, string? key, string message)
        : base(message)
    {
        Before = before;
        After = after;
        Key = key;
    }

    /// <summary>The timestamp before the gap.</summary>
    public string Before { get; }

    /// <summary>The timestamp after the gap.</summary>
    public string After { get; }

    /// <summary>The series key when scanning a population, or null.</summary>
    public string? Key { get; }
}
