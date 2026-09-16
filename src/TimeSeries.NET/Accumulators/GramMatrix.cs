namespace TimeSeries.Accumulators;

/// <summary>
/// A frozen lag-augmented Gram matrix <c>G = sum_t v_t v_t'</c> over
/// <c>v_t = (y_t, ..., y_(t-L))</c>, together with the column sums and row count that
/// accompany it.
/// </summary>
/// <remarks>
/// <para>
/// This is the single summary both Hannan-Rissanen stages are computed from. The pilot
/// AR(m) normal equations are the submatrix <c>G[1..m, 1..m]</c> with right-hand side
/// <c>G[1..m, 0]</c>; the stage-two equations are <c>A'GA</c> and <c>A'G e0</c> for a
/// projection <c>A</c> built from the pilot coefficients alone. Neither needs the series.
/// </para>
/// <para>
/// <c>G</c> is not Toeplitz: <c>G[0,1]</c> and <c>G[1,2]</c> span the same lag distance
/// but cover row ranges offset by one. Anything needing a genuinely Toeplitz sequence —
/// Durbin-Levinson, and therefore the PACF — must use
/// <see cref="Autocovariances"/> instead.
/// </para>
/// </remarks>
public sealed class GramMatrix
{
    private readonly double[] _values;
    private readonly double[] _sums;

    internal GramMatrix(int lagDepth, double[] values, double[] sums, double count, long rows, double offset)
    {
        LagDepth = lagDepth;
        _values = values;
        _sums = sums;
        Count = count;
        Rows = rows;
        Offset = offset;
    }

    /// <summary>The deepest lag included, <c>L</c>.</summary>
    public int LagDepth { get; }

    /// <summary>The side length of the matrix, <c>L + 1</c>.</summary>
    public int Dimension => LagDepth + 1;

    /// <summary>
    /// The number of complete lag rows folded in, <c>N = n - L</c>, after any forgetting factor.
    /// </summary>
    public double Count { get; }

    /// <summary>The number of rows added, unaffected by any forgetting factor.</summary>
    public long Rows { get; }

    /// <summary>
    /// The constant subtracted from every value before accumulation, to keep
    /// <c>sum x^2</c> from swamping the deviations it is meant to measure. Coefficients
    /// solved from this matrix are on the shifted scale and must be un-shifted, which for
    /// a model with an intercept means the intercept absorbs it.
    /// </summary>
    public double Offset { get; }

    /// <summary>The full symmetric matrix, row-major, <c>Dimension * Dimension</c> values.</summary>
    public ReadOnlyMemory<double> Values => _values;

    /// <summary>
    /// Column sums <c>s = sum_t v_t</c>, on the shifted scale. These are what an intercept
    /// term is built from.
    /// </summary>
    public ReadOnlyMemory<double> Sums => _sums;

    /// <summary>Reads one entry of the matrix.</summary>
    /// <param name="i">Row index, a lag in <c>[0, L]</c>.</param>
    /// <param name="j">Column index, a lag in <c>[0, L]</c>.</param>
    /// <returns>The accumulated cross-product <c>sum_t y_(t-i) y_(t-j)</c> on the shifted scale.</returns>
    public double this[int i, int j] => _values[(i * Dimension) + j];
}
