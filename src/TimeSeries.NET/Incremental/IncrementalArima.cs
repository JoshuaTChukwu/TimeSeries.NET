using TimeSeries.Data;

namespace TimeSeries;

/// <summary>
/// An estimator whose state persists between folds. Because the Gram matrix, sums and
/// count are pure sums, folding new observations into saved state gives a result
/// identical to refitting from scratch over the union — batch and incremental are the
/// same computation reached by different routes.
/// </summary>
/// <remarks>
/// <para>
/// Not thread-safe: one instance per series, folds in sequence. The state serialises to
/// a few kilobytes — <c>(D+1)^2 + D + 2</c> doubles plus the differencing windows and a
/// bounded tail — small enough to live in a row of the consumer's own database next to
/// the data it summarises.
/// </para>
/// <para>
/// With <see cref="ArimaOptions.ForgettingFactor"/> below 1 the state is scaled by that
/// factor before each fold, ageing earlier folds out at a chosen half-life. This is a
/// modelling choice for shifting regimes, not a numerical necessity; the pilot and ARMA
/// coefficients are re-derived from the updated state at every solve, so there is no
/// frozen pilot and no accumulated drift to compensate for.
/// </para>
/// </remarks>
public sealed class IncrementalArima
{
    private const string Magic = "TSIA";
    private readonly ArimaScan _scan;

    /// <summary>
    /// Creates an estimator with empty state.
    /// </summary>
    /// <param name="options">The model. Validated immediately.</param>
    /// <param name="regressorCount">Exogenous regressors per observation; zero for ARIMA.</param>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The options are inconsistent, or <paramref name="regressorCount"/> is negative.</exception>
    public IncrementalArima(ArimaOptions options, int regressorCount = 0)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (regressorCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(regressorCount), regressorCount, "Regressor count must be zero or greater.");
        }

        options.Validate();
        Options = options;
        RegressorCount = regressorCount;
        _scan = new ArimaScan(options, regressorCount);
    }

    /// <summary>The model being estimated.</summary>
    public ArimaOptions Options { get; }

    /// <summary>Exogenous regressors per observation.</summary>
    public int RegressorCount { get; }

    /// <summary>
    /// The rows folded so far, as <c>[0, count)</c>. Its <see cref="FitWindow.To"/> is the
    /// ordinal the next fold should begin at.
    /// </summary>
    public FitWindow Watermark => new(0, _scan.RawCount);

    /// <summary>How many folds have been applied.</summary>
    public int FoldCount { get; private set; }

    /// <summary>
    /// Folds new observations into the state.
    /// </summary>
    /// <param name="values">The new observations, continuing the series in time order.</param>
    /// <exception cref="InvalidOperationException">The model has regressors; use the overload that takes them.</exception>
    /// <exception cref="InvalidSeriesException">A value is NaN or infinite.</exception>
    public void Fold(ReadOnlySpan<double> values)
    {
        if (RegressorCount > 0)
        {
            throw new InvalidOperationException(
                $"The model has {RegressorCount} regressor(s); fold with their values as well.");
        }

        BeginFold();
        AcceptInBatches(values, ReadOnlySpan<double>.Empty);
    }

    /// <summary>
    /// Folds new observations and their regressors into the state.
    /// </summary>
    /// <param name="values">The new observations, continuing the series in time order.</param>
    /// <param name="exogenous">Regressors aligned row for row with <paramref name="values"/>.</param>
    /// <exception cref="ArgumentNullException"><paramref name="exogenous"/> is null.</exception>
    /// <exception cref="ArgumentException">The regressors do not align with the values or the model.</exception>
    /// <exception cref="InvalidSeriesException">A value is NaN or infinite.</exception>
    public void Fold(ReadOnlySpan<double> values, ExogenousMatrix exogenous)
    {
        if (exogenous is null)
        {
            throw new ArgumentNullException(nameof(exogenous));
        }

        if (exogenous.RegressorCount != RegressorCount)
        {
            throw new ArgumentException(
                $"The model has {RegressorCount} regressor(s) but {exogenous.RegressorCount} were supplied.", nameof(exogenous));
        }

        if (exogenous.Count != values.Length)
        {
            throw new ArgumentException(
                $"{values.Length} observations but {exogenous.Count} regressor rows; they must align.", nameof(exogenous));
        }

        BeginFold();
        AcceptInBatches(values, exogenous.Values.Span);
    }

    /// <summary>
    /// Folds every batch a source yields into the state — typically the rows since the
    /// last <see cref="Watermark"/>.
    /// </summary>
    /// <param name="newRows">The new rows, continuing the series in time order.</param>
    /// <param name="cancellationToken">Checked at every batch boundary.</param>
    /// <returns>A task that completes when the source is exhausted and disposed.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="newRows"/> is null.</exception>
    /// <exception cref="ArgumentException">The source's regressor count differs from the model's.</exception>
    public async ValueTask FoldAsync(ITimeSeriesSource newRows, CancellationToken cancellationToken = default)
    {
        if (newRows is null)
        {
            throw new ArgumentNullException(nameof(newRows));
        }

        if (newRows.RegressorCount != RegressorCount)
        {
            throw new ArgumentException(
                $"The model has {RegressorCount} regressor(s) but the source supplies {newRows.RegressorCount}.", nameof(newRows));
        }

        BeginFold();
        var cursor = await newRows.OpenAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            while (await cursor.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                var batch = cursor.Current;
                _scan.Accept(batch.Values.Span, batch.Exogenous.Span);
            }
        }
        finally
        {
            await cursor.DisposeAsync().ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Solves from the current state. May be called after every fold; it reads no data
    /// and leaves the state untouched.
    /// </summary>
    /// <returns>The model fitted to everything folded so far.</returns>
    /// <exception cref="InsufficientDataException">Too few observations have been folded.</exception>
    /// <exception cref="SingularDesignException">The normal equations are singular for a non-constant series.</exception>
    public ArimaFit Solve() => _scan.Solve(Watermark);

    /// <summary>
    /// Writes the complete state so estimation can resume later, from another process,
    /// with exactly the same result as if it had never stopped.
    /// </summary>
    /// <param name="destination">A writable stream. Left open.</param>
    /// <exception cref="ArgumentNullException"><paramref name="destination"/> is null.</exception>
    public void SaveTo(Stream destination)
    {
        if (destination is null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        using var writer = new BinaryWriter(destination, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(Magic);
        writer.Write(RegressorCount);
        writer.Write(FoldCount);
        _scan.WriteState(writer);
        writer.Flush();
    }

    /// <summary>
    /// Restores an estimator from state written by <see cref="SaveTo"/>.
    /// </summary>
    /// <param name="state">The saved state. Left open.</param>
    /// <param name="options">The model the state was saved for; must match.</param>
    /// <returns>An estimator ready for its next fold.</returns>
    /// <exception cref="ArgumentNullException">Either argument is null.</exception>
    /// <exception cref="InvalidDataException">The stream is not a saved state, or was saved for a different model.</exception>
    public static IncrementalArima Restore(Stream state, ArimaOptions options)
    {
        if (state is null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        using var reader = new BinaryReader(state, System.Text.Encoding.UTF8, leaveOpen: true);
        var magic = reader.ReadString();

        if (!string.Equals(magic, Magic, StringComparison.Ordinal))
        {
            throw new InvalidDataException("The stream does not hold IncrementalArima state.");
        }

        var regressorCount = reader.ReadInt32();
        var folds = reader.ReadInt32();
        var estimator = new IncrementalArima(options, regressorCount) { FoldCount = folds };
        estimator._scan.ReadState(reader);
        return estimator;
    }

    private void BeginFold()
    {
        if (FoldCount > 0 && Options.ForgettingFactor < 1d)
        {
            _scan.Scale(Options.ForgettingFactor);
        }

        FoldCount++;
    }

    private void AcceptInBatches(ReadOnlySpan<double> values, ReadOnlySpan<double> exogenous)
    {
        for (var start = 0; start < values.Length; start += ArimaModel.BatchSize)
        {
            var take = Math.Min(ArimaModel.BatchSize, values.Length - start);
            var rows = exogenous.IsEmpty
                ? ReadOnlySpan<double>.Empty
                : exogenous.Slice(start * RegressorCount, take * RegressorCount);

            _scan.Accept(values.Slice(start, take), rows);
        }
    }
}
