using TimeSeries.Accumulators;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Accumulators;

public class AutocovarianceAccumulatorTests
{
    [Theory]
    [InlineData(0.7)]
    [InlineData(0.4)]
    [InlineData(-0.6)]
    public void Ar1_HasAutocorrelationPhiToThePowerK(double phi)
    {
        // The closed-form identity for AR(1). It needs no reference implementation and no
        // committed constants: rho_k = phi^k is what the process is.
        var series = SeriesGenerator.Ar1(200_000, phi, seed: 2024);

        var acf = new AccumulatorScan(lagDepth: 10).Scan(series).Autocovariance.Freeze().Acf.Span;

        Assert.Equal(1d, acf[0], tolerance: 1e-12);

        for (var k = 1; k <= 10; k++)
        {
            Assert.Equal(Math.Pow(phi, k), acf[k], tolerance: 0.02);
        }
    }

    [Fact]
    public void Ar1_VarianceMatchesTheTheoreticalSigmaSquaredOverOneMinusPhiSquared()
    {
        const double Phi = 0.7;
        const double Sigma = 1.5;
        var series = SeriesGenerator.Ar1(200_000, Phi, seed: 2025, Sigma);

        var gamma0 = new AccumulatorScan(lagDepth: 5).Scan(series).Autocovariance.Freeze().Variance;
        var expected = (Sigma * Sigma) / (1d - (Phi * Phi));

        Assert.Equal(expected, gamma0, tolerance: expected * 0.03);
    }

    [Fact]
    public void Ma1_HasAutocorrelationThetaOverOnePlusThetaSquared_AndZeroBeyondLagOne()
    {
        const double Theta = 0.6;
        var series = SeriesGenerator.Ma1(200_000, Theta, seed: 2026);

        var acf = new AccumulatorScan(lagDepth: 6).Scan(series).Autocovariance.Freeze().Acf.Span;

        Assert.Equal(Theta / (1d + (Theta * Theta)), acf[1], tolerance: 0.02);

        for (var k = 2; k <= 6; k++)
        {
            Assert.Equal(0d, acf[k], tolerance: 0.02);
        }
    }

    [Fact]
    public void MatchesTheTextbookDivideByNEstimatorComputedInTwoPasses()
    {
        var series = SeriesGenerator.Ar1(3000, phi: 0.55, seed: 2027);
        const int MaxLag = 12;

        var actual = new AccumulatorScan(MaxLag).Scan(series).Autocovariance.Freeze();
        var expected = Naive.Autocovariance(series, MaxLag);

        Assert.Equal(series.Length, actual.Count);

        for (var k = 0; k <= MaxLag; k++)
        {
            Assert.Equal(expected[k], actual.Gamma.Span[k], tolerance: Math.Abs(expected[k]) * 1e-10);
        }
    }

    [Fact]
    public void AutocorrelationsStayInsideTheUnitInterval_AsAPositiveSemiDefiniteSequenceMust()
    {
        // The property the divide-by-n estimator was chosen for. A divide-by-(n-k)
        // estimator can and does break it, and Durbin-Levinson then walks off the rails.
        for (var seed = 1; seed <= 25; seed++)
        {
            var series = SeriesGenerator.Ar1(400, phi: 0.95, seed);
            var acf = new AccumulatorScan(lagDepth: 60).Scan(series).Autocovariance.Freeze().Acf.Span;

            for (var k = 0; k < acf.Length; k++)
            {
                Assert.InRange(acf[k], -1d, 1d);
            }
        }
    }

    [Fact]
    public void TheOffsetKeepsTheEstimatesAccurateWhenTheLevelDwarfsTheDeviations()
    {
        // Spec section 9's hazard made concrete: a level near 40,000 carrying deviations
        // near 1. Accumulating sum(x^2) at that level spends most of a double's
        // significant digits representing the level, and the variance is what is left
        // over. The offset is subtracted before anything is squared, which is
        // algebraically a no-op and numerically the whole game.
        var series = SeriesGenerator.Ar1(2000, phi: 0.5, seed: 2028);
        for (var i = 0; i < series.Length; i++)
        {
            series[i] += 40_000d;
        }

        const int MaxLag = 8;

        // Centred first, so the reference never forms a large sum of squares at all.
        var reference = Naive.Autocovariance(series, MaxLag);

        var unshifted = new AccumulatorScan(MaxLag).Scan(series).Autocovariance.Freeze();
        var shifted = new AccumulatorScan(MaxLag, offset: series[0]).Scan(series)
            .Autocovariance.Freeze();

        Assert.Equal(unshifted.Mean, shifted.Mean, tolerance: 1e-6);

        for (var k = 0; k <= MaxLag; k++)
        {
            Assert.Equal(reference[k], shifted.Gamma.Span[k], tolerance: Math.Abs(reference[k]) * 1e-9);
        }

        var shiftedError = Math.Abs(shifted.Variance - reference[0]);
        var unshiftedError = Math.Abs(unshifted.Variance - reference[0]);

        Assert.True(
            unshiftedError > shiftedError * 100d,
            $"Expected the offset to buy at least two orders of magnitude of accuracy, but " +
            $"the unshifted error was {unshiftedError:E3} against {shiftedError:E3}.");
    }

    [Fact]
    public void PartitionedScanWithWarmup_EqualsASinglePass_BitForBit()
    {
        var series = new double[1500];
        var random = new Random(2029);
        for (var i = 0; i < series.Length; i++)
        {
            series[i] = random.Next(-500, 501);
        }

        const int MaxLag = 7;
        var single = new AccumulatorScan(MaxLag).Scan(series).Autocovariance.Freeze();

        var merged = new AccumulatorScan(MaxLag).ScanRange(series, 0, 500);
        merged.Merge(new AccumulatorScan(MaxLag).ScanRange(series, 500, 1000));
        merged.Merge(new AccumulatorScan(MaxLag).ScanRange(series, 1000, 1500));

        var actual = merged.Autocovariance.Freeze();

        Assert.Equal(single.Count, actual.Count);
        Assert.Equal(single.Mean, actual.Mean);
        Assert.Equal(single.Gamma.ToArray(), actual.Gamma.ToArray());
    }

    [Fact]
    public void EarlyObservationsContributeOnlyToTheLagsTheyHavePartnersFor()
    {
        // gamma_k sums over t in [k+1, n], so a five-point series contributes five terms
        // at lag 0, four at lag 1, and so on. Getting this wrong is the classic way to
        // break the estimator's positive semi-definiteness.
        double[] series = [1, 2, 3, 4, 5];
        const int MaxLag = 3;

        var actual = new AccumulatorScan(MaxLag).Scan(series).Autocovariance.Freeze();
        var expected = Naive.Autocovariance(series, MaxLag);

        Assert.Equal(5d, actual.Count);
        Assert.Equal(3d, actual.Mean, tolerance: 1e-12);

        for (var k = 0; k <= MaxLag; k++)
        {
            Assert.Equal(expected[k], actual.Gamma.Span[k], tolerance: 1e-12);
        }
    }

    [Theory]
    [InlineData(0d)]
    [InlineData(7d)]
    [InlineData(1234.5678d)]
    [InlineData(-98765.4321d)]
    [InlineData(40_000d)]
    [InlineData(1e8)]
    public void AConstantSeriesHasZeroVarianceAtEveryLevel(double constant)
    {
        // Not a formality. Expanding the centred sum leaves a difference of nearly equal
        // large numbers, and at some levels — 1234.5678 among them — the residue is
        // around 1e-8 rather than zero. Divided by itself that residue becomes an
        // autocorrelation near 0.98, an invented signal from a series that never moves.
        var series = new double[500];
        Array.Fill(series, constant);

        var result = new AccumulatorScan(lagDepth: 4).Scan(series).Autocovariance.Freeze();

        Assert.Equal(0d, result.Variance);
        Assert.Equal(constant, result.Mean, tolerance: 1e-9);
        Assert.Equal(1d, result.Acf.Span[0]);

        for (var k = 1; k <= 4; k++)
        {
            Assert.Equal(0d, result.Gamma.Span[k]);
            Assert.Equal(0d, result.Acf.Span[k]);
        }
    }

    [Fact]
    public void Freeze_IsPureAndRepeatable()
    {
        var series = SeriesGenerator.Ar1(500, phi: 0.6, seed: 2030);
        var accumulator = new AccumulatorScan(lagDepth: 5).Scan(series).Autocovariance;

        var first = accumulator.Freeze();
        var second = accumulator.Freeze();

        Assert.Equal(first.Gamma.ToArray(), second.Gamma.ToArray());
        Assert.False(first.Gamma.Equals(second.Gamma));
    }

    [Fact]
    public void Scale_ReweightsEveryObservationEqually_SoTheEstimatesDoNotMove()
    {
        var series = SeriesGenerator.Ar1(1000, phi: 0.5, seed: 2031);
        var accumulator = new AccumulatorScan(lagDepth: 5).Scan(series).Autocovariance;

        var before = accumulator.Freeze();
        accumulator.Scale(0.25);
        var after = accumulator.Freeze();

        // The weight falls, and that is all that falls. Unlike the Gram, which exposes raw
        // cross-products, the autocovariances are a weighted average: the sums and the
        // counts are scaled together, so the factor cancels. Ageing only shows up once
        // newer observations are folded in at full weight against the discounted history.
        Assert.Equal(before.Count * 0.25, after.Count, tolerance: 1e-12);
        Assert.Equal(before.Mean, after.Mean, tolerance: 1e-9);

        for (var k = 0; k <= 5; k++)
        {
            Assert.Equal(before.Gamma.Span[k], after.Gamma.Span[k], tolerance: Math.Abs(before.Gamma.Span[k]) * 1e-12);
            Assert.Equal(before.Acf.Span[k], after.Acf.Span[k], tolerance: 1e-12);
        }
    }

    [Fact]
    public void ScaleThenFold_PullsTheEstimateTowardsTheNewerObservations()
    {
        // What the forgetting factor is actually for. The older block is discounted, the
        // newer one arrives at full weight, and the mean lands nearer the newer block than
        // an even-handed pass over both would put it.
        var older = SeriesGenerator.Ar1(1000, phi: 0.5, seed: 2032);
        var newer = SeriesGenerator.Ar1(1000, phi: 0.5, seed: 2033);
        for (var i = 0; i < newer.Length; i++)
        {
            newer[i] += 10d;
        }

        var combined = older.Concat(newer).ToArray();
        var unweighted = new AccumulatorScan(lagDepth: 3).Scan(combined).Autocovariance.Freeze();

        var scan = new AccumulatorScan(lagDepth: 3);
        scan.Scan(older);
        scan.Autocovariance.Scale(0.1);
        scan.Scan(newer);
        var aged = scan.Autocovariance.Freeze();

        var newerMean = newer.Average();

        Assert.True(
            Math.Abs(aged.Mean - newerMean) < Math.Abs(unweighted.Mean - newerMean),
            $"Aged mean {aged.Mean:F4} should sit closer to the newer block's mean " +
            $"{newerMean:F4} than the unweighted mean {unweighted.Mean:F4} does.");
    }

    [Fact]
    public void MergingMismatchedState_Throws()
    {
        var target = new AutocovarianceAccumulator(maxLag: 4, offset: 2d);

        Assert.Throws<ArgumentException>(() => target.Merge(new AutocovarianceAccumulator(5, 2d)));
        Assert.Throws<ArgumentException>(() => target.Merge(new AutocovarianceAccumulator(4, 3d)));
    }

    [Fact]
    public void NegativeMaxLag_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new AutocovarianceAccumulator(-1));
    }
}
