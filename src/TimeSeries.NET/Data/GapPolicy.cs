namespace TimeSeries.Data;

/// <summary>
/// What to do when two consecutive observations are further apart than the declared
/// step. ARIMA assumes regular spacing, so a gap has to be resolved one way or another
/// — and never silently.
/// </summary>
public enum GapPolicy
{
    /// <summary>
    /// Refuse: throw <see cref="SeriesGapException"/> naming the timestamps either side.
    /// The default, because a silently misaligned fit is the worst outcome.
    /// </summary>
    Throw,

    /// <summary>
    /// Repeat the last observation (and its regressors) once per missing step.
    /// Fabricates observations the model then treats as real; use knowingly.
    /// </summary>
    ForwardFill,

    /// <summary>
    /// Fill missing steps by linear interpolation between the observations either side.
    /// Smoother than <see cref="ForwardFill"/>, but it leaks future information backwards
    /// and biases autocovariances downward; use knowingly.
    /// </summary>
    Interpolate,
}
