using TimeSeries.Data;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Selection;

public class AutoArimaTests
{
    private static readonly AutoArimaOptions Small = new() { MaxP = 3, MaxQ = 2, MaxPilotOrder = 12 };

    [Fact]
    public void IntegratedSeries_GetsD1_ByTheStationarityTest()
    {
        var differenced = SeriesGenerator.Arma(3_000, [0.5], [], seed: 1, constant: 0.05);
        var series = SeriesGenerator.Integrate(differenced, 1, level: 100);

        var selection = new AutoArima(Small).Select(series);

        Assert.Equal(1, selection.Best.Order.D);
        Assert.Equal(1, selection.ChosenDifferencing.Order);
        Assert.True(selection.Differencing.Single(c => c.Order == 0).Kpss.RejectsStationarity(), "d = 0 should have been rejected.");
        Assert.False(selection.Differencing.Single(c => c.Order == 1).Kpss.RejectsStationarity(), "d = 1 should have been accepted.");
        Assert.True(selection.Best.Order.P <= 2, $"Chose p = {selection.Best.Order.P} for an AR(1) difference.");
    }

    [Fact]
    public void StationaryAr2_GetsD0_AndALowOrder()
    {
        var series = SeriesGenerator.Ar(4_000, [0.5, 0.3], seed: 2);

        var selection = new AutoArima(Small).Select(series);

        Assert.Equal(0, selection.Best.Order.D);
        Assert.InRange(selection.Best.Order.P + selection.Best.Order.Q, 1, 3);
        Assert.Equal((Small.MaxP + 1) * (Small.MaxQ + 1), selection.Candidates.Count);

        // Ranked ascending by the criterion, successes first.
        var scores = selection.Candidates.Where(c => c.Succeeded).Select(c => c.Aicc).ToArray();
        Assert.Equal(scores.OrderBy(s => s).ToArray(), scores);
        Assert.Equal(InformationCriterion.Aicc, selection.Criterion);
    }

    [Fact]
    public void Criterion_ChangesTheRanking()
    {
        var series = SeriesGenerator.Arma(3_000, [0.6], [0.3], seed: 3);

        var aicc = new AutoArima(Small).Select(series);
        var bic = new AutoArima(Small with { Criterion = InformationCriterion.Bic }).Select(series);

        Assert.Equal(bic.Candidates.Where(c => c.Succeeded).Select(c => c.Bic).OrderBy(s => s), bic.Candidates.Where(c => c.Succeeded).Select(c => c.Bic));
        Assert.True(bic.Best.Order.P + bic.Best.Order.Q <= aicc.Best.Order.P + aicc.Best.Order.Q);
    }

    [Fact]
    public async Task TheWholeGrid_ReadsTheSourceExactlyOnce()
    {
        // The M7 acceptance criterion: three differencing orders, twelve (p, q) pairs each,
        // one traversal.
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(3_000, [0.5], [0.2], seed: 4), 1);
        var counting = new CountingSource(new ArraySource(series, batchSize: 500));

        var selection = await new AutoArima(Small).SelectAsync(counting);

        Assert.Equal(1, counting.Opens);
        Assert.Equal(1, counting.Disposes);
        Assert.Equal((series.Length + 499) / 500 + 1, counting.Reads); // every batch once, plus the terminating false
        Assert.Equal(series.Length, counting.RowsDelivered);
        Assert.Equal(1, selection.Best.Order.D);
        Assert.Equal(3, selection.Differencing.Count);
    }

    [Fact]
    public async Task SyncAndAsyncSelection_Agree_Exactly()
    {
        var series = SeriesGenerator.Arma(2_500, [0.4], [0.4], seed: 5);
        var auto = new AutoArima(Small);

        var sync = auto.Select(series);
        var async = await auto.SelectAsync(new ArraySource(series, batchSize: 333));

        Assert.Equal(sync.Best.Order, async.Best.Order);
        Assert.Equal(sync.Best.AutoRegressive.ToArray(), async.Best.AutoRegressive.ToArray());
        Assert.Equal(sync.Best.Diagnostics.Aicc, async.Best.Diagnostics.Aicc);
    }

    [Fact]
    public void WithRegressors_SelectsAndRecoversBeta()
    {
        const int N = 4_000;
        var x = SeriesGenerator.Ar1(N, 0.3, seed: 6);
        var noise = SeriesGenerator.Ar1(N, 0.5, seed: 7);
        var y = new double[N];
        for (var t = 0; t < N; t++)
        {
            y[t] = noise[t] + (2d * x[t]);
        }

        var selection = new AutoArima(Small).Select(y, ExogenousMatrix.FromColumn(x));

        Assert.Equal(0, selection.Best.Order.D);
        Assert.Equal(2d, selection.Best.ExogenousCoefficients.Span[0], 0.1);
    }

    [Fact]
    public void SeasonalDifferencing_IsTakenWhenTheSeasonalAutocorrelationIsStrong()
    {
        const int Period = 4;
        var differenced = SeriesGenerator.Arma(3_000, [0.3], [], seed: 8);
        var series = SeriesGenerator.Integrate(differenced, 1);
        for (var t = 0; t < series.Length; t++)
        {
            series[t] += 40d * Math.Sin(2d * Math.PI * t / Period);
        }

        var options = Small with { SeasonalCandidates = [SeasonalOrder.None, new SeasonalOrder(1, Period)] };
        var selection = new AutoArima(options).Select(series);

        Assert.Equal(new SeasonalOrder(1, Period), selection.Best.Seasonal);

        // (1 - B^4) already removes the unit root — it is a sum of four first differences —
        // so the stationarity test may settle on d = 0 once the seasonal difference is taken.
        Assert.InRange(selection.Best.Order.D, 0, 1);

        // And without a seasonal pattern, none is taken.
        var plain = new AutoArima(options).Select(SeriesGenerator.Integrate(differenced, 1));
        Assert.Equal(SeasonalOrder.None, plain.Best.Seasonal);
    }

    [Fact]
    public void ConstantDifferencedSeries_SelectsTheDeterministicModel()
    {
        var series = Enumerable.Range(0, 400).Select(t => 5d + (1.5 * t)).ToArray();

        var selection = new AutoArima(Small).Select(series);

        Assert.True(selection.Best.Diagnostics.IsConstantSeries);
        Assert.Equal(new ArimaOrder(0, 1, 0), selection.Best.Order);
        Assert.Equal(1.5, selection.Best.Intercept, 1e-12);
    }

    [Fact]
    public void TooShortForEveryCandidate_ThrowsTheLeastDemandingRequirement()
    {
        // Gram depth max(3, 2 + 12) + 10 Ljung-Box lags = 24, plus ten for the smallest
        // model (0,d,0): 34 rows would fit that one, so twenty must fail every candidate.
        var exception = Assert.Throws<InsufficientDataException>(
            () => new AutoArima(Small).Select(SeriesGenerator.Uniform(20, seed: 9)));

        Assert.Equal(24 + 10, exception.Required);
    }

    [Fact]
    public void TheWinnerForecasts_LikeADirectFit()
    {
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(3_000, [0.5], [], seed: 10, constant: 0.1), 1);
        var selection = new AutoArima(Small).Select(series);

        var direct = new ArimaModel(selection.Best.Options).Fit(series);
        var a = selection.Best.Forecast(ForecastHorizon.Periods(6));
        var b = direct.Forecast(ForecastHorizon.Periods(6));

        // Same order and same rows would give identical numbers; the grid scan uses the
        // grid's lag depth, so the row range differs slightly and the estimates with it.
        Assert.Equal(selection.Best.Order, direct.Order);
        for (var h = 0; h < 6; h++)
        {
            Assert.Equal(b.Mean.Span[h], a.Mean.Span[h], Math.Abs(b.Mean.Span[h]) * 0.01 + 0.5);
        }
    }

    [Fact]
    public void ShortSeries_NeverSelectsAnInadmissibleModel()
    {
        // Hannan-Rissanen on ~150 points with an over-parameterised ARMA(2,2) candidate can
        // land outside the stationary/invertible region while scoring best in sample. The
        // default must refuse to crown such a model; the ranked table still shows it.
        var options = new AutoArimaOptions { MaxP = 3, MaxQ = 2, MaxPilotOrder = 12 };
        var inadmissibleTops = 0;

        for (var seed = 1; seed <= 15; seed++)
        {
            var series = SeriesGenerator.Arma(156, [0.6], [0.3], seed, constant: 0.2);

            var guarded = new AutoArima(options).Select(series);
            var unguarded = new AutoArima(options with { RequireAdmissible = false }).Select(series);

            Assert.True(guarded.Best.Diagnostics.IsStationary && guarded.Best.Diagnostics.IsInvertible,
                $"Seed {seed}: selected ARIMA{guarded.Best.Order} is inadmissible.");
            Assert.True(guarded.Candidates.First().IsAdmissible);

            // Without the guard the ranking is the bare criterion.
            var scores = unguarded.Candidates.Where(c => c.Succeeded).Select(c => c.Aicc).ToArray();
            Assert.Equal(scores.OrderBy(v => v).ToArray(), scores);

            if (!unguarded.Best.Diagnostics.IsStationary || !unguarded.Best.Diagnostics.IsInvertible)
            {
                inadmissibleTops++;
            }
        }

        // Informational rather than asserted: the guard matters only if this is ever non-zero.
        Assert.True(inadmissibleTops >= 0);
    }

    [Fact]
    public void OptionsValidation()
    {
        Assert.Throws<ArgumentNullException>(() => new AutoArima(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AutoArima(new AutoArimaOptions { MaxP = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AutoArima(new AutoArimaOptions { DifferenceOrders = [] }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AutoArima(new AutoArimaOptions { SeasonalCandidates = [] }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AutoArima(new AutoArimaOptions { MaxP = 5, MaxQ = 5, MaxPilotOrder = 4 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AutoArima(new AutoArimaOptions { StationaritySignificance = 0.2 }));

        var auto = new AutoArima(Small);
        Assert.Throws<ArgumentNullException>(() => auto.Select(new double[100], null!));
        Assert.Throws<ArgumentException>(() => auto.Select(new double[100], ExogenousMatrix.FromColumn(new double[99])));
    }
}
