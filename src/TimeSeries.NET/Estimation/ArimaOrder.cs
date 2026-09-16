namespace TimeSeries;

/// <summary>The non-seasonal order <c>(p, d, q)</c> of an ARIMA model.</summary>
/// <param name="P">Autoregressive order.</param>
/// <param name="D">Differencing order.</param>
/// <param name="Q">Moving-average order.</param>
public readonly record struct ArimaOrder(int P, int D, int Q)
{
    /// <summary>A readable form such as <c>(2,1,1)</c>.</summary>
    /// <returns>The order in the conventional parenthesised notation.</returns>
    public override string ToString() => $"({P},{D},{Q})";
}

/// <summary>
/// Seasonal differencing <c>(1 - B^s)^D</c>. Seasonal AR and MA terms are deferred to a
/// later version; only the differencing ships in v1.
/// </summary>
/// <param name="D">Seasonal differencing order.</param>
/// <param name="Period">Seasonal period <c>s</c>; at least 2 when <paramref name="D"/> is positive.</param>
public readonly record struct SeasonalOrder(int D, int Period)
{
    /// <summary>No seasonal differencing.</summary>
    public static readonly SeasonalOrder None = new(0, 0);

    /// <summary>A readable form such as <c>(1)[12]</c>, or <c>none</c>.</summary>
    /// <returns>The seasonal order and period.</returns>
    public override string ToString() => D == 0 ? "none" : $"({D})[{Period}]";
}
