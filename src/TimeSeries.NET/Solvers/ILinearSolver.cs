namespace TimeSeries.Solvers;

/// <summary>
/// Solves the normal equations <c>G x = b</c> for a symmetric positive semi-definite
/// <c>G</c>. The seam behind which the numerical backend can change without the
/// estimator noticing.
/// </summary>
/// <remarks>
/// A singular system is reported as a <c>false</c> return with diagnostics, not as an
/// exception thrown from four layers down. The estimator decides what a singular design
/// means for the caller — usually a constant differenced series, which has its own
/// short circuit — and what to suggest.
/// </remarks>
public interface ILinearSolver
{
    /// <summary>
    /// Attempts to solve <c>G x = b</c>.
    /// </summary>
    /// <param name="gram">
    /// The matrix <c>G</c>, row-major, <c>n * n</c> values where <c>n</c> is the length of
    /// <paramref name="rhs"/>.
    /// </param>
    /// <param name="rhs">The right-hand side <c>b</c>, <c>n</c> values.</param>
    /// <param name="coefficients">Receives <c>x</c>. At least <c>n</c> values.</param>
    /// <param name="diagnostics">How the solve went, whether or not it succeeded.</param>
    /// <returns>True when a solution was produced; false when the system is singular.</returns>
    /// <exception cref="ArgumentException">
    /// The spans are inconsistently sized.
    /// </exception>
    bool TrySolve(
        ReadOnlySpan<double> gram,
        ReadOnlySpan<double> rhs,
        Span<double> coefficients,
        out SolveDiagnostics diagnostics);
}
