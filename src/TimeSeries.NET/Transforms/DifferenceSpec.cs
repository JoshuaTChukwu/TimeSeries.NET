namespace TimeSeries.Transforms;

/// <summary>
/// The differencing applied to a series before estimation: <c>(1-B)^d (1-B^s)^D</c>.
/// </summary>
/// <remarks>
/// <para>
/// This is the single description of differencing in the library. Non-seasonal order
/// <c>d</c>, seasonal order <c>D</c> and seasonal period <c>s</c> travel together so a
/// transform, an integration state and a fitted model cannot disagree about what was
/// differenced.
/// </para>
/// <para>
/// The two operators commute, so the library applies the seasonal differences first and
/// the non-seasonal differences second. That order is fixed rather than incidental: it
/// determines the layout of <see cref="IntegrationState"/>.
/// </para>
/// </remarks>
public readonly record struct DifferenceSpec
{
    /// <summary>No differencing: the series passes through unchanged.</summary>
    public static readonly DifferenceSpec None = default;

    /// <summary>
    /// Creates a differencing specification.
    /// </summary>
    /// <param name="order">Non-seasonal differencing order <c>d</c>. Zero or greater.</param>
    /// <param name="seasonalOrder">Seasonal differencing order <c>D</c>. Zero or greater.</param>
    /// <param name="period">
    /// Seasonal period <c>s</c>. Must be 2 or greater when <paramref name="seasonalOrder"/>
    /// is positive; normalised to zero when it is not, so that two specifications
    /// describing the same transform compare equal.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// An order is negative, or a seasonal difference is requested with a period below 2.
    /// </exception>
    public DifferenceSpec(int order, int seasonalOrder = 0, int period = 0)
    {
        if (order < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(order), order, "Differencing order must be zero or greater.");
        }

        if (seasonalOrder < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(seasonalOrder), seasonalOrder,
                "Seasonal differencing order must be zero or greater.");
        }

        if (period < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(period), period, "Seasonal period must be zero or greater.");
        }

        if (seasonalOrder > 0 && period < 2)
        {
            throw new ArgumentOutOfRangeException(
                nameof(period), period,
                "Seasonal period must be 2 or greater when a seasonal difference is requested. " +
                "A period of 1 is a non-seasonal difference; raise 'order' instead.");
        }

        Order = order;
        SeasonalOrder = seasonalOrder;
        Period = seasonalOrder > 0 ? period : 0;
    }

    /// <summary>Non-seasonal differencing order <c>d</c>.</summary>
    public int Order { get; }

    /// <summary>Seasonal differencing order <c>D</c>.</summary>
    public int SeasonalOrder { get; }

    /// <summary>Seasonal period <c>s</c>, or zero when <see cref="SeasonalOrder"/> is zero.</summary>
    public int Period { get; }

    /// <summary>
    /// Observations consumed before the transform produces its first output,
    /// <c>d + D * s</c>. This is also the length of the tail an
    /// <see cref="IntegrationState"/> carries, and the warm-up prefix a partitioned
    /// reader must read before its own range.
    /// </summary>
    public int WarmupLength => Order + (SeasonalOrder * Period);

    /// <summary>True when no differencing is applied.</summary>
    public bool IsIdentity => Order == 0 && SeasonalOrder == 0;

    /// <summary>A readable form such as <c>(1-B)^1(1-B^12)^1</c>.</summary>
    /// <returns>The differencing operator in polynomial notation.</returns>
    public override string ToString()
    {
        if (IsIdentity)
        {
            return "identity";
        }

        var text = string.Empty;
        if (Order > 0)
        {
            text += $"(1-B)^{Order}";
        }

        if (SeasonalOrder > 0)
        {
            text += $"(1-B^{Period})^{SeasonalOrder}";
        }

        return text;
    }
}
