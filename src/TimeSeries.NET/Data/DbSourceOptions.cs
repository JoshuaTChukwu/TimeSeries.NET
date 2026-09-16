namespace TimeSeries.Data;

/// <summary>
/// How a database-backed source treats spacing and ordering. Shared by every adapter —
/// SQL through <c>DbDataReader</c>, MongoDB, and any contributed source that assembles its
/// rows through <see cref="SeriesBatchAssembler"/> — so a gap means the same thing
/// whichever store the series lives in.
/// </summary>
public sealed record DbSourceOptions
{
    /// <summary>The defaults: gaps throw, key order is validated.</summary>
    public static DbSourceOptions Default { get; } = new();

    /// <summary>What to do at a gap. Default <see cref="GapPolicy.Throw"/>.</summary>
    public GapPolicy Gaps { get; init; } = GapPolicy.Throw;

    /// <summary>
    /// The nominal spacing between observations for a date/time (or ISO text) time column.
    /// Gap detection is active only when this or <see cref="ExpectedNumericStep"/> is set.
    /// </summary>
    public TimeSpan? ExpectedStep { get; init; }

    /// <summary>The nominal spacing for a numeric time column, in its own units.</summary>
    public double? ExpectedNumericStep { get; init; }

    /// <summary>
    /// The largest spacing that is <em>not</em> a gap. Defaults to the expected step.
    /// Business-daily data, for instance, declares a one-day step and a four-day maximum
    /// so weekends pass and anything longer is caught.
    /// </summary>
    public TimeSpan? MaxGap { get; init; }

    /// <summary>The largest spacing that is not a gap, for a numeric time column.</summary>
    public double? MaxNumericGap { get; init; }

    /// <summary>
    /// Whether to throw when a key is smaller than the one before it in a grouped scan.
    /// Default true. Turn off only when the database collation cannot be matched by
    /// <see cref="KeyComparer"/>.
    /// </summary>
    public bool ValidateKeyOrder { get; init; } = true;

    /// <summary>
    /// How keys are compared for order validation. Null compares strings ordinally and
    /// other comparable types by their own ordering.
    /// </summary>
    public IComparer<object>? KeyComparer { get; init; }

    internal double? Step => ExpectedStep?.Ticks ?? ExpectedNumericStep;

    internal double? MaximumGap => MaxGap?.Ticks ?? MaxNumericGap ?? Step;
}
