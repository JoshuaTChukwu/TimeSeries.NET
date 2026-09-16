namespace TimeSeries.Solvers;

/// <summary>
/// Decides whether every root of <c>1 + a_1 z + ... + a_n z^n</c> lies outside the unit
/// circle, without computing the roots.
/// </summary>
/// <remarks>
/// <para>
/// This is the Schur-Cohn test in the form of the Levinson step-down recursion: peel the
/// polynomial down one degree at a time, and the process is minimum-phase exactly when
/// every reflection coefficient met along the way has magnitude below 1. It is exact,
/// dependency-free and <c>O(n^2)</c>.
/// </para>
/// <para>
/// Applied to <c>theta(z) = 1 + theta_1 z + ... + theta_q z^q</c> it is the invertibility
/// check on a moving-average part; applied to <c>phi(z) = 1 - phi_1 z - ... - phi_p z^p</c>
/// it is the stationarity check on an autoregressive part.
/// </para>
/// </remarks>
public static class PolynomialStability
{
    /// <summary>
    /// Tests whether all roots of <c>1 + a_1 z + ... + a_n z^n</c> lie strictly outside the
    /// unit circle.
    /// </summary>
    /// <param name="coefficients"><c>a_1 .. a_n</c>. Empty means the constant polynomial 1, which passes.</param>
    /// <param name="margin">
    /// Receives <c>1 - max |k_i|</c> over the reflection coefficients: how far inside the
    /// stable region the polynomial sits. Zero or negative when it fails.
    /// </param>
    /// <returns>True when minimum-phase.</returns>
    public static bool IsMinimumPhase(ReadOnlySpan<double> coefficients, out double margin)
    {
        var n = coefficients.Length;
        margin = 1d;

        if (n == 0)
        {
            return true;
        }

        Span<double> current = n <= 64 ? stackalloc double[n] : new double[n];
        Span<double> next = n <= 64 ? stackalloc double[n] : new double[n];
        coefficients.CopyTo(current);

        for (var degree = n; degree >= 1; degree--)
        {
            var reflection = current[degree - 1];
            var magnitude = Math.Abs(reflection);
            margin = Math.Min(margin, 1d - magnitude);

            if (magnitude >= 1d)
            {
                return false;
            }

            // Step down: a'_i = (a_i - k a_(m-i)) / (1 - k^2) for i = 1 .. m-1.
            var denominator = 1d - (reflection * reflection);

            for (var i = 1; i < degree; i++)
            {
                next[i - 1] = (current[i - 1] - (reflection * current[degree - 1 - i])) / denominator;
            }

            next.Slice(0, degree - 1).CopyTo(current);
        }

        return true;
    }

    /// <summary>
    /// Tests whether the autoregressive polynomial <c>1 - phi_1 z - ... - phi_p z^p</c> is
    /// stationary — all roots outside the unit circle.
    /// </summary>
    /// <param name="phi">The autoregressive coefficients <c>phi_1 .. phi_p</c>.</param>
    /// <param name="margin">How far inside the stationary region the polynomial sits.</param>
    /// <returns>True when stationary.</returns>
    public static bool IsStationary(ReadOnlySpan<double> phi, out double margin)
    {
        Span<double> negated = phi.Length <= 64 ? stackalloc double[phi.Length] : new double[phi.Length];

        for (var i = 0; i < phi.Length; i++)
        {
            negated[i] = -phi[i];
        }

        return IsMinimumPhase(negated, out margin);
    }

    /// <summary>
    /// Tests whether the moving-average polynomial <c>1 + theta_1 z + ... + theta_q z^q</c>
    /// is invertible — all roots outside the unit circle.
    /// </summary>
    /// <param name="theta">The moving-average coefficients <c>theta_1 .. theta_q</c>, additive convention.</param>
    /// <param name="margin">How far inside the invertible region the polynomial sits.</param>
    /// <returns>True when invertible.</returns>
    public static bool IsInvertible(ReadOnlySpan<double> theta, out double margin)
        => IsMinimumPhase(theta, out margin);
}
