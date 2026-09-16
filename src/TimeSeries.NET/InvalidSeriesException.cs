namespace TimeSeries;

/// <summary>
/// A series or regressor contains a value the library cannot compute with — NaN or an
/// infinity — at a specific position.
/// </summary>
/// <remarks>
/// Thrown rather than skipped, per the validation contract in the specification: a fit
/// that silently drops or imputes an observation is a fit over a series the caller did
/// not supply.
/// </remarks>
public sealed class InvalidSeriesException : ArgumentException
{
    /// <summary>Creates the exception for a value at <paramref name="index"/>.</summary>
    /// <param name="paramName">The argument the series arrived in.</param>
    /// <param name="index">The zero-based position of the offending value.</param>
    /// <param name="value">The offending value.</param>
    public InvalidSeriesException(string paramName, long index, double value)
        : base($"Series value at index {index} is {value}; every value must be finite.", paramName)
    {
        Index = index;
        Value = value;
    }

    /// <summary>The zero-based position of the offending value.</summary>
    public long Index { get; }

    /// <summary>The offending value.</summary>
    public double Value { get; }
}
