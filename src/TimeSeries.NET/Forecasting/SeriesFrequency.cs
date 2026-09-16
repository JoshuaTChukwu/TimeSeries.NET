namespace TimeSeries;

/// <summary>
/// The spacing of a regularly observed series, declared so that a calendar horizon such
/// as "two years" can be resolved to a number of steps rather than guessed.
/// </summary>
public enum SeriesFrequency
{
    /// <summary>One observation per year.</summary>
    Yearly = 1,

    /// <summary>One observation per quarter.</summary>
    Quarterly = 4,

    /// <summary>One observation per calendar month.</summary>
    Monthly = 12,

    /// <summary>One observation per week.</summary>
    Weekly = 52,

    /// <summary>One observation per calendar day.</summary>
    Daily = 365,

    /// <summary>One observation per business day, taken as 252 a year.</summary>
    BusinessDaily = 252,

    /// <summary>One observation per hour.</summary>
    Hourly = 8760,
}
