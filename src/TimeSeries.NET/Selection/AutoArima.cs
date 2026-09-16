using TimeSeries.Data;

namespace TimeSeries;

/// <summary>
/// Chooses an ARIMA order automatically, in one pass over the data.
/// </summary>
/// <remarks>
/// <para>
/// Each candidate differencing gets its own scan — a different transform produces a
/// different series — but all of them are fed from the same batches, so the source is
/// read once. Within a scan every <c>(p, q)</c> is a projection of the same matrix and
/// costs no data access at all.
/// </para>
/// <para>
/// <c>d</c> is chosen by a KPSS test, never by information criterion; see
/// <see cref="AutoArimaOptions"/>. <c>(p, q)</c> are then ranked by the criterion at that
/// fixed differencing, where the comparison is valid because every candidate is fitted to
/// the same rows.
/// </para>
/// </remarks>
public sealed class AutoArima
{
    /// <summary>
    /// Creates a selector.
    /// </summary>
    /// <param name="options">The search space. Validated immediately.</param>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is null.</exception>
    public AutoArima(AutoArimaOptions options)
    {
        Options = options ?? throw new ArgumentNullException(nameof(options));
        options.Validate();
    }

    /// <summary>The search space.</summary>
    public AutoArimaOptions Options { get; }

    /// <summary>
    /// Selects and fits the best model for an in-memory series.
    /// </summary>
    /// <param name="series">The series, in time order.</param>
    /// <returns>The winner and the ranked table.</returns>
    public ArimaSelection Select(ReadOnlySpan<double> series)
    {
        var scans = BuildScans(regressorCount: 0);

        for (var start = 0; start < series.Length; start += ArimaModel.BatchSize)
        {
            var take = Math.Min(ArimaModel.BatchSize, series.Length - start);
            Feed(scans, series.Slice(start, take), []);
        }

        return Choose(scans, new FitWindow(0, series.Length));
    }

    /// <summary>
    /// Selects and fits the best model for an in-memory series with regressors.
    /// </summary>
    /// <param name="series">The series, in time order.</param>
    /// <param name="exogenous">Regressors aligned row for row with <paramref name="series"/>.</param>
    /// <returns>The winner and the ranked table.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="exogenous"/> is null.</exception>
    /// <exception cref="ArgumentException">The regressors do not align with the series.</exception>
    public ArimaSelection Select(ReadOnlySpan<double> series, ExogenousMatrix exogenous)
    {
        if (exogenous is null)
        {
            throw new ArgumentNullException(nameof(exogenous));
        }

        if (exogenous.Count != series.Length)
        {
            throw new ArgumentException(
                $"The series has {series.Length} observations but the regressors have {exogenous.Count} rows.", nameof(exogenous));
        }

        var scans = BuildScans(exogenous.RegressorCount);

        for (var start = 0; start < series.Length; start += ArimaModel.BatchSize)
        {
            var take = Math.Min(ArimaModel.BatchSize, series.Length - start);
            Feed(scans, series.Slice(start, take), exogenous.Rows(start, take));
        }

        return Choose(scans, new FitWindow(0, series.Length));
    }

    /// <summary>
    /// Selects and fits the best model from a source, reading it exactly once for the
    /// whole grid.
    /// </summary>
    /// <param name="source">Where the series comes from.</param>
    /// <param name="cancellationToken">Checked at every batch boundary.</param>
    /// <returns>The winner and the ranked table.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    public async ValueTask<ArimaSelection> SelectAsync(ITimeSeriesSource source, CancellationToken cancellationToken = default)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        var scans = BuildScans(source.RegressorCount);
        var cursor = await source.OpenAsync(cancellationToken).ConfigureAwait(false);
        long rows = 0;

        try
        {
            while (await cursor.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                var batch = cursor.Current;
                Feed(scans, batch.Values.Span, batch.Exogenous.Span);
                rows += batch.Count;
            }
        }
        finally
        {
            await cursor.DisposeAsync().ConfigureAwait(false);
        }

        return Choose(scans, new FitWindow(0, rows));
    }

    private sealed record Scan(int Order, SeasonalOrder Seasonal, ArimaScan Estimator);

    private List<Scan> BuildScans(int regressorCount)
    {
        var maxPeriod = Options.SeasonalCandidates.Max(s => s.Period);
        var stationarityLags = Math.Max(32, maxPeriod);
        var scans = new List<Scan>();

        foreach (var seasonal in Options.SeasonalCandidates)
        {
            foreach (var d in Options.DifferenceOrders)
            {
                scans.Add(new Scan(d, seasonal, new ArimaScan(Options.Widest(d, seasonal), regressorCount, stationarityLags)));
            }
        }

        return scans;
    }

    private static void Feed(List<Scan> scans, ReadOnlySpan<double> values, ReadOnlySpan<double> exogenous)
    {
        foreach (var scan in scans)
        {
            scan.Estimator.Accept(values, exogenous);
        }
    }

    private ArimaSelection Choose(List<Scan> scans, FitWindow window)
    {
        var maxPeriod = Options.SeasonalCandidates.Max(s => s.Period);
        var tested = new List<DifferencingCandidate>();
        var byKey = new Dictionary<(int, SeasonalOrder), (Scan Scan, DifferencingCandidate Candidate)>();

        foreach (var scan in scans)
        {
            var kpss = scan.Estimator.Stationarity!;
            var moments = scan.Estimator.Moments;
            var seasonalAcf = maxPeriod > 0 && maxPeriod <= kpss.Autocovariances.MaxLag && kpss.Autocovariances.Variance > 0d
                ? kpss.Autocovariances.Acf.Span[maxPeriod]
                : double.NaN;
            var candidate = new DifferencingCandidate(
                scan.Order, scan.Seasonal, kpss, seasonalAcf, moments.Count > 0d && moments.Maximum.Equals(moments.Minimum));

            tested.Add(candidate);
            byKey[(scan.Order, scan.Seasonal)] = (scan, candidate);
        }

        // d for each seasonal candidate: the first that is not rejected, else the largest.
        var ascending = Options.DifferenceOrders.OrderBy(d => d).ToArray();
        var chosenBySeasonal = new Dictionary<SeasonalOrder, DifferencingCandidate>();

        foreach (var seasonal in Options.SeasonalCandidates)
        {
            DifferencingCandidate? chosen = null;

            foreach (var d in ascending)
            {
                var candidate = byKey[(d, seasonal)].Candidate;

                if (candidate.IsConstant || !candidate.Kpss.RejectsStationarity(Options.StationaritySignificance))
                {
                    chosen = candidate;
                    break;
                }
            }

            chosenBySeasonal[seasonal] = chosen ?? byKey[(ascending[ascending.Length - 1], seasonal)].Candidate;
        }

        // Seasonal differencing: the first candidate whose remaining seasonal autocorrelation
        // is below the threshold, else the last. With one candidate there is no choice.
        var chosenDifferencing = chosenBySeasonal[Options.SeasonalCandidates[Options.SeasonalCandidates.Count - 1]];

        foreach (var seasonal in Options.SeasonalCandidates)
        {
            var candidate = chosenBySeasonal[seasonal];

            if (double.IsNaN(candidate.SeasonalAutocorrelation)
                || Math.Abs(candidate.SeasonalAutocorrelation) < Options.SeasonalStrengthThreshold)
            {
                chosenDifferencing = candidate;
                break;
            }
        }

        var winnerScan = byKey[(chosenDifferencing.Order, chosenDifferencing.Seasonal)].Scan.Estimator;
        var candidates = new List<ArimaCandidate>();

        for (var p = 0; p <= Options.MaxP; p++)
        {
            for (var q = 0; q <= Options.MaxQ; q++)
            {
                var options = Options.Candidate(p, chosenDifferencing.Order, q, chosenDifferencing.Seasonal);

                try
                {
                    candidates.Add(new ArimaCandidate(options.Order, options.Seasonal, winnerScan.Solve(options, window), null));
                }
                catch (InsufficientDataException exception)
                {
                    candidates.Add(new ArimaCandidate(options.Order, options.Seasonal, null, exception));
                }
                catch (SingularDesignException exception)
                {
                    candidates.Add(new ArimaCandidate(options.Order, options.Seasonal, null, exception));
                }
            }
        }

        var ranked = candidates
            .OrderBy(c => c.Succeeded ? 0 : 1)
            .ThenBy(c => c.Succeeded && double.IsNaN(c.Score(Options.Criterion)) ? 1 : 0)
            .ThenBy(c => c.Score(Options.Criterion))
            .ThenBy(c => c.Order.P + c.Order.Q)
            .ToList();

        var best = ranked[0];

        if (!best.Succeeded)
        {
            // Nothing fitted. Surface the least demanding failure so the message is actionable.
            var insufficient = candidates.Select(c => c.Failure).OfType<InsufficientDataException>().OrderBy(e => e.Required).FirstOrDefault();
            throw insufficient ?? candidates[0].Failure!;
        }

        return new ArimaSelection(best.Fit!, ranked, tested, chosenDifferencing, Options.Criterion);
    }
}
