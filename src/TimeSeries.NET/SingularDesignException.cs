namespace TimeSeries;

/// <summary>
/// The normal equations are singular for a reason other than a constant differenced
/// series — typically two regressors that are collinear, or a lag structure the data
/// cannot identify. The message suggests a ridge.
/// </summary>
public sealed class SingularDesignException : InvalidOperationException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="stage">Which estimation stage met the singular system.</param>
    /// <param name="column">The design column at which elimination found no pivot.</param>
    /// <param name="message">Diagnosis and a suggested remedy.</param>
    public SingularDesignException(string stage, int column, string message)
        : base(message)
    {
        Stage = stage;
        Column = column;
    }

    /// <summary>Which estimation stage met the singular system.</summary>
    public string Stage { get; }

    /// <summary>The design column at which elimination found no pivot.</summary>
    public int Column { get; }
}
