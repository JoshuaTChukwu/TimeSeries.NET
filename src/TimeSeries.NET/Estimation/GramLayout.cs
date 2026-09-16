namespace TimeSeries;

/// <summary>
/// Where each quantity sits in the lag-augmented Gram row
/// <c>v_t = (z_t, z_(t-1), ..., z_(t-L), x_t, x_(t-1), ..., x_(t-q))</c>, and in the
/// matrix augmented by a constant column.
/// </summary>
/// <remarks>
/// The exogenous block carries lags <c>0..q</c> of every regressor because the pilot
/// residual <c>e-hat_t</c> includes <c>b' x_t</c>, so its own lags <c>e-hat_(t-j)</c>
/// involve <c>x_(t-j)</c>. Keeping those lags in the row is what lets the second stage
/// remain an exact linear functional of one accumulated matrix.
/// </remarks>
internal readonly struct GramLayout
{
    internal GramLayout(int lagDepth, int regressorCount, int exogenousLags)
    {
        LagDepth = lagDepth;
        RegressorCount = regressorCount;
        ExogenousLags = exogenousLags;
    }

    /// <summary><c>L</c>: the deepest lag of the series in the row.</summary>
    internal int LagDepth { get; }

    /// <summary><c>r</c>: regressors per observation.</summary>
    internal int RegressorCount { get; }

    /// <summary><c>q</c>: the deepest exogenous lag in the row.</summary>
    internal int ExogenousLags { get; }

    /// <summary>The row length, <c>L + 1 + r (q + 1)</c>.</summary>
    internal int Dimension => LagDepth + 1 + (RegressorCount * (ExogenousLags + 1));

    /// <summary>The index of the constant in the augmented matrix.</summary>
    internal int Constant => Dimension;

    /// <summary>The side of the augmented matrix, <c>Dimension + 1</c>.</summary>
    internal int Augmented => Dimension + 1;

    /// <summary>The index of <c>z_(t-k)</c>.</summary>
    internal int Lag(int k) => k;

    /// <summary>The index of regressor <paramref name="regressor"/> at lag <paramref name="lag"/>.</summary>
    internal int Exogenous(int lag, int regressor) => LagDepth + 1 + (lag * RegressorCount) + regressor;
}
