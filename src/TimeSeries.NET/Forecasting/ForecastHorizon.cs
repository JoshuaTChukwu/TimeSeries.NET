namespace TimeSeries;

/// <summary>The unit a <see cref="ForecastHorizon"/> is expressed in.</summary>
public enum HorizonUnit
{
    /// <summary>Steps at the series' own frequency. Needs no declared frequency.</summary>
    Periods,

    /// <summary>Calendar days.</summary>
    Days,

    /// <summary>Calendar months.</summary>
    Months,

    /// <summary>Calendar years.</summary>
    Years,
}

/// <summary>
/// How far ahead to forecast, in periods or in calendar time. Calendar horizons resolve
/// against the model's declared <see cref="SeriesFrequency"/> and throw rather than guess
/// when there is none.
/// </summary>
public readonly record struct ForecastHorizon
{
    private ForecastHorizon(HorizonUnit unit, double amount)
    {
        Unit = unit;
        Amount = amount;
    }

    /// <summary>The unit.</summary>
    public HorizonUnit Unit { get; }

    /// <summary>The amount, in <see cref="Unit"/>.</summary>
    public double Amount { get; }

    /// <summary>A horizon of <paramref name="count"/> steps at the series' own frequency.</summary>
    /// <param name="count">Steps ahead, at least 1.</param>
    /// <returns>The horizon.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is below 1.</exception>
    public static ForecastHorizon Periods(int count) => new(HorizonUnit.Periods, Positive(count, nameof(count)));

    /// <summary>A horizon of <paramref name="count"/> calendar days.</summary>
    /// <param name="count">Days ahead, positive.</param>
    /// <returns>The horizon.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is not positive.</exception>
    public static ForecastHorizon Days(int count) => new(HorizonUnit.Days, Positive(count, nameof(count)));

    /// <summary>A horizon of <paramref name="count"/> calendar months.</summary>
    /// <param name="count">Months ahead, positive.</param>
    /// <returns>The horizon.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is not positive.</exception>
    public static ForecastHorizon Months(int count) => new(HorizonUnit.Months, Positive(count, nameof(count)));

    /// <summary>A horizon of <paramref name="count"/> years, fractional allowed.</summary>
    /// <param name="count">Years ahead, positive.</param>
    /// <returns>The horizon.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is not positive or not finite.</exception>
    public static ForecastHorizon Years(double count)
    {
        if (!(count > 0d) || double.IsInfinity(count))
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, "Years must be positive and finite.");
        }

        return new(HorizonUnit.Years, count);
    }

    /// <summary>
    /// Resolves the horizon to a number of steps.
    /// </summary>
    /// <param name="frequency">The series' declared frequency, or null when none was declared.</param>
    /// <returns>Steps ahead, at least 1.</returns>
    /// <exception cref="ForecastHorizonException">
    /// A calendar horizon with no declared frequency, or one that rounds to fewer than one step.
    /// </exception>
    public int Resolve(SeriesFrequency? frequency)
    {
        if (Unit == HorizonUnit.Periods)
        {
            return (int)Amount;
        }

        if (frequency is not SeriesFrequency declared)
        {
            throw new ForecastHorizonException(
                $"The horizon {this} is in calendar time but the model declares no SeriesFrequency, so it " +
                "cannot be resolved to a number of periods. Set ArimaOptions.Frequency, or use " +
                "ForecastHorizon.Periods.");
        }

        var perYear = (double)(int)declared;
        var years = Unit switch
        {
            HorizonUnit.Years => Amount,
            HorizonUnit.Months => Amount / 12d,
            HorizonUnit.Days => Amount / 365d,
            _ => throw new InvalidOperationException($"Unknown horizon unit {Unit}."),
        };

        var steps = (int)Math.Round(years * perYear, MidpointRounding.AwayFromZero);

        if (steps < 1)
        {
            throw new ForecastHorizonException(
                $"The horizon {this} resolves to {years * perYear:0.###} periods at {declared} frequency, " +
                "which rounds to fewer than one step.");
        }

        return steps;
    }

    /// <summary>A readable form such as <c>2 years</c> or <c>12 periods</c>.</summary>
    /// <returns>The horizon in words.</returns>
    public override string ToString() => $"{Amount:0.###} {Unit.ToString().ToLowerInvariant()}";

    private static int Positive(int value, string name)
    {
        if (value < 1)
        {
            throw new ArgumentOutOfRangeException(name, value, "The horizon must be at least 1.");
        }

        return value;
    }
}
