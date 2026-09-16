namespace TimeSeries.Solvers;

/// <summary>
/// What happened inside a linear solve — enough to tell a healthy system from one that
/// was close to singular, and to say where a singular one failed.
/// </summary>
public readonly record struct SolveDiagnostics
{
    /// <summary>The number of unknowns.</summary>
    public int Dimension { get; init; }

    /// <summary>True when a solution was produced.</summary>
    public bool Succeeded { get; init; }

    /// <summary>
    /// The smallest pivot met during elimination, on the scaled system. Near the pivot
    /// tolerance means the design is close to singular.
    /// </summary>
    public double SmallestPivot { get; init; }

    /// <summary>The largest pivot met during elimination, on the scaled system.</summary>
    public double LargestPivot { get; init; }

    /// <summary>
    /// <see cref="LargestPivot"/> over <see cref="SmallestPivot"/>: a cheap stand-in for
    /// the condition number. Infinite when the solve failed.
    /// </summary>
    public double PivotRatio =>
        SmallestPivot > 0d ? LargestPivot / SmallestPivot : double.PositiveInfinity;

    /// <summary>The ridge added to the scaled diagonal, zero when none.</summary>
    public double Ridge { get; init; }

    /// <summary>True when the columns were scaled to unit diagonal before elimination.</summary>
    public bool ColumnsScaled { get; init; }

    /// <summary>
    /// The column at which elimination found no usable pivot, or -1 when the solve succeeded.
    /// </summary>
    public int FailedColumn { get; init; }
}
