namespace TimeSeries;

/// <summary>
/// The series is too short for the model requested. Thrown rather than fitting
/// something: a coefficient table from twenty observations is noise with decimal places.
/// </summary>
public sealed class InsufficientDataException : InvalidOperationException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="required">The minimum number of usable observations for this model.</param>
    /// <param name="available">The number of usable observations supplied.</param>
    /// <param name="message">What "usable" means here and what to do about it.</param>
    public InsufficientDataException(long required, long available, string message)
        : base(message)
    {
        Required = required;
        Available = available;
    }

    /// <summary>The minimum number of usable observations for this model.</summary>
    public long Required { get; }

    /// <summary>The number of usable observations supplied.</summary>
    public long Available { get; }
}
