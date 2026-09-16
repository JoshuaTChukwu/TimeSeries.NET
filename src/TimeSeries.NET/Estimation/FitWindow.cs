namespace TimeSeries;

/// <summary>
/// The half-open row range <c>[From, To)</c> a model was fitted over. For an in-memory
/// fit these are array indices; for a streamed fit they are whatever ordinal the source
/// assigns, and the incremental estimator uses <see cref="To"/> as its watermark.
/// </summary>
/// <param name="From">The first row included.</param>
/// <param name="To">One past the last row included.</param>
public readonly record struct FitWindow(long From, long To)
{
    /// <summary>The number of rows in the window.</summary>
    public long Count => To - From;

    /// <summary>A readable form such as <c>[0, 10000)</c>.</summary>
    /// <returns>The range in interval notation.</returns>
    public override string ToString() => $"[{From}, {To})";
}
