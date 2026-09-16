namespace TimeSeries.Extensions.DependencyInjection;

/// <summary>
/// Creates estimators from options at runtime — for callers whose model orders are
/// configuration rather than code, or who want one seam to substitute in tests.
/// </summary>
public interface IArimaModelFactory
{
    /// <summary>Creates a stateless, thread-safe estimator for a fixed order.</summary>
    /// <param name="options">The model.</param>
    /// <returns>The estimator.</returns>
    ArimaModel Create(ArimaOptions options);

    /// <summary>Creates an automatic order selector.</summary>
    /// <param name="options">The search space.</param>
    /// <returns>The selector.</returns>
    AutoArima CreateAutoArima(AutoArimaOptions options);

    /// <summary>Creates an incremental estimator with empty state. One per series; not thread-safe.</summary>
    /// <param name="options">The model.</param>
    /// <param name="regressorCount">Exogenous regressors per observation.</param>
    /// <returns>The estimator.</returns>
    IncrementalArima CreateIncremental(ArimaOptions options, int regressorCount = 0);

    /// <summary>Restores an incremental estimator from saved state.</summary>
    /// <param name="state">State written by <see cref="IncrementalArima.SaveTo"/>.</param>
    /// <param name="options">The model the state was saved for.</param>
    /// <returns>The estimator, ready for its next fold.</returns>
    IncrementalArima RestoreIncremental(Stream state, ArimaOptions options);
}

/// <summary>The default factory: constructs the library types directly.</summary>
public sealed class ArimaModelFactory : IArimaModelFactory
{
    /// <inheritdoc />
    public ArimaModel Create(ArimaOptions options) => new(options);

    /// <inheritdoc />
    public AutoArima CreateAutoArima(AutoArimaOptions options) => new(options);

    /// <inheritdoc />
    public IncrementalArima CreateIncremental(ArimaOptions options, int regressorCount = 0) => new(options, regressorCount);

    /// <inheritdoc />
    public IncrementalArima RestoreIncremental(Stream state, ArimaOptions options) => IncrementalArima.Restore(state, options);
}
