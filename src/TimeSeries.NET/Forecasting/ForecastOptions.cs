namespace TimeSeries;

/// <summary>Choices about how a forecast is produced. Immutable; the default is usually right.</summary>
public sealed record ForecastOptions
{
    /// <summary>The defaults: 95% intervals, non-invertible models warned about rather than refused.</summary>
    public static ForecastOptions Default { get; } = new();

    /// <summary>The confidence level of the prediction intervals, strictly inside (0, 1). Default 0.95.</summary>
    public double Confidence { get; init; } = 0.95;

    /// <summary>
    /// Whether to refuse to forecast from a model whose moving-average part is not
    /// invertible. Default false: the forecast is produced and
    /// <see cref="ForecastResult.Warnings"/> says why its intervals should not be trusted.
    /// </summary>
    public bool RequireInvertible { get; init; }

    /// <summary>Checks the options.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="Confidence"/> is outside (0, 1).</exception>
    public void Validate()
    {
        if (!(Confidence > 0d && Confidence < 1d))
        {
            throw new ArgumentOutOfRangeException(nameof(Confidence), Confidence, "Confidence must lie strictly inside (0, 1).");
        }
    }
}
