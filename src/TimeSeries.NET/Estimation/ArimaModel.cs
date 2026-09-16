namespace TimeSeries;

/// <summary>
/// Fits ARIMA and ARIMAX models. Stateless and thread-safe: one instance can fit any
/// number of series concurrently, and every fit returns an independent
/// <see cref="ArimaFit"/>.
/// </summary>
/// <remarks>
/// One class covers both ARIMA and ARIMAX — they are the same model, and the presence of
/// exogenous regressors is the only difference. Estimation is by Hannan-Rissanen in a
/// single pass over the data; see <see cref="ArimaOptions"/> for the model form.
/// </remarks>
public sealed partial class ArimaModel
{
    /// <summary>Rows folded into the accumulators per step. Matches the streaming batch size.</summary>
    internal const int BatchSize = 8192;

    /// <summary>
    /// Creates an estimator for the given model.
    /// </summary>
    /// <param name="options">The model to fit. Validated immediately.</param>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The options are inconsistent; see <see cref="ArimaOptions.Validate"/>.</exception>
    public ArimaModel(ArimaOptions options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        options.Validate();
        Options = options;
    }

    /// <summary>The model this estimator fits.</summary>
    public ArimaOptions Options { get; }

    /// <summary>
    /// Fits an ARIMA model to an in-memory series.
    /// </summary>
    /// <param name="series">The series, in time order.</param>
    /// <returns>The fitted model.</returns>
    /// <exception cref="InvalidSeriesException">The series contains NaN or an infinity.</exception>
    /// <exception cref="InsufficientDataException">The series is too short for the model.</exception>
    /// <exception cref="SingularDesignException">The normal equations are singular for a non-constant series.</exception>
    public ArimaFit Fit(ReadOnlySpan<double> series)
    {
        var scan = new ArimaScan(Options, regressorCount: 0);

        for (var start = 0; start < series.Length; start += BatchSize)
        {
            var take = Math.Min(BatchSize, series.Length - start);
            scan.Accept(series.Slice(start, take), []);
        }

        return scan.Solve(new FitWindow(0, series.Length));
    }

    /// <summary>
    /// Fits an ARIMAX model to an in-memory series with exogenous regressors.
    /// </summary>
    /// <param name="series">The series, in time order.</param>
    /// <param name="exogenous">Regressors aligned row for row with <paramref name="series"/>.</param>
    /// <returns>The fitted model.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="exogenous"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="exogenous"/> has a different number of rows than the series.</exception>
    /// <exception cref="InvalidSeriesException">The series or a regressor contains NaN or an infinity.</exception>
    /// <exception cref="InsufficientDataException">The series is too short for the model.</exception>
    /// <exception cref="SingularDesignException">The normal equations are singular for a non-constant series.</exception>
    public ArimaFit Fit(ReadOnlySpan<double> series, ExogenousMatrix exogenous)
    {
        if (exogenous is null)
        {
            throw new ArgumentNullException(nameof(exogenous));
        }

        if (exogenous.Count != series.Length)
        {
            throw new ArgumentException(
                $"The series has {series.Length} observations but the regressors have {exogenous.Count} rows; " +
                "they must align row for row.",
                nameof(exogenous));
        }

        var scan = new ArimaScan(Options, exogenous.RegressorCount);

        for (var start = 0; start < series.Length; start += BatchSize)
        {
            var take = Math.Min(BatchSize, series.Length - start);
            scan.Accept(series.Slice(start, take), exogenous.Rows(start, take));
        }

        return scan.Solve(new FitWindow(0, series.Length));
    }
}
