using TimeSeries.Solvers;
using TimeSeries.Statistics;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Solvers;

public class DurbinLevinsonTests
{
    /// <summary>Theoretical autocorrelations of AR(2) by the Yule-Walker recursion.</summary>
    private static double[] Ar2Autocorrelations(double phi1, double phi2, int maxLag)
    {
        var rho = new double[maxLag + 1];
        rho[0] = 1d;
        rho[1] = phi1 / (1d - phi2);

        for (var k = 2; k <= maxLag; k++)
        {
            rho[k] = (phi1 * rho[k - 1]) + (phi2 * rho[k - 2]);
        }

        return rho;
    }

    [Theory]
    [InlineData(0.9)]
    [InlineData(0.5)]
    [InlineData(-0.7)]
    public void Ar1_TheoreticalSequence_PacfIsPhiThenExactlyZero(double phi)
    {
        // rho_k = phi^k. PACF cuts off after lag 1 in exact arithmetic.
        var rho = new double[11];
        for (var k = 0; k <= 10; k++)
        {
            rho[k] = Math.Pow(phi, k);
        }

        var pacf = DurbinLevinson.Pacf(rho, 10);

        Assert.Equal(phi, pacf[0], 1e-14);
        for (var k = 1; k < 10; k++)
        {
            Assert.Equal(0d, pacf[k], 1e-12);
        }
    }

    [Fact]
    public void Ar2_TheoreticalSequence_PacfCutsOffAfterLagTwo()
    {
        const double Phi1 = 0.5;
        const double Phi2 = 0.3;
        var rho = Ar2Autocorrelations(Phi1, Phi2, 12);

        var pacf = DurbinLevinson.Pacf(rho, 12);

        Assert.Equal(5d / 7d, pacf[0], 1e-14);   // phi_11 = rho_1 = phi1 / (1 - phi2)
        Assert.Equal(Phi2, pacf[1], 1e-14);       // phi_22 = phi2
        for (var k = 2; k < 12; k++)
        {
            Assert.Equal(0d, pacf[k], 1e-12);
        }
    }

    [Fact]
    public void Ar2_YuleWalker_RecoversCoefficientsAndInnovationVariance()
    {
        const double Phi1 = 0.5;
        const double Phi2 = 0.3;
        var rho = Ar2Autocorrelations(Phi1, Phi2, 4);
        var coefficients = new double[2];

        Assert.True(DurbinLevinson.TryYuleWalker(rho, 2, coefficients, out var innovationVariance));

        Assert.Equal(Phi1, coefficients[0], 1e-14);
        Assert.Equal(Phi2, coefficients[1], 1e-14);

        // sigma^2 / gamma_0 = 1 - phi1 rho_1 - phi2 rho_2, and equals prod (1 - phi_kk^2).
        var expected = 1d - (Phi1 * rho[1]) - (Phi2 * rho[2]);
        Assert.Equal(expected, innovationVariance, 1e-14);
        Assert.Equal((1d - (25d / 49d)) * (1d - (Phi2 * Phi2)), innovationVariance, 1e-14);
    }

    [Fact]
    public void SimulatedAr2_PacfCutsOffWithinWhiteNoiseBounds()
    {
        const int N = 20_000;
        var series = SeriesGenerator.Ar(N, [0.5, 0.3], seed: 2024);

        var pacf = Pacf.Compute(series, 10);

        // Four standard errors: a sampling test that must never flake.
        var bound = 4d / Math.Sqrt(N);

        Assert.Equal(5d / 7d, pacf[0], 0.03);
        Assert.Equal(0.3, pacf[1], 0.03);
        for (var k = 2; k < 10; k++)
        {
            Assert.True(Math.Abs(pacf[k]) < bound, $"PACF at lag {k + 1} is {pacf[k]}, beyond ±{bound}.");
        }
    }

    [Theory]
    [InlineData(200, 100)]
    [InlineData(500, 250)]
    [InlineData(64, 63)]
    public void BiasedAutocovariances_AreAlwaysPositiveSemiDefinite(int length, int maxLag)
    {
        // The reason the divide-by-n estimator was chosen: Durbin-Levinson never meets a
        // non-positive innovation variance on it, even at lags close to the series length
        // and on a series as persistent as a random walk.
        foreach (var seed in new[] { 1, 2, 3, 4, 5 })
        {
            var noise = SeriesGenerator.Uniform(length, seed);
            var walk = SeriesGenerator.RandomWalk(length, seed, drift: 0.1);

            var pacfNoise = new double[maxLag];
            var pacfWalk = new double[maxLag];

            Assert.True(DurbinLevinson.TryPacf(Acf.Compute(noise, maxLag).Gamma.Span, maxLag, pacfNoise, out _));
            Assert.True(DurbinLevinson.TryPacf(Acf.Compute(walk, maxLag).Gamma.Span, maxLag, pacfWalk, out _));

            Assert.All(pacfNoise, value => Assert.True(Math.Abs(value) <= 1d));
            Assert.All(pacfWalk, value => Assert.True(Math.Abs(value) <= 1d));
        }
    }

    [Fact]
    public void PerfectlyPredictableSequence_IsReportedDegenerate()
    {
        // gamma_k = gamma_0 for all k: phi_11 = 1, innovation variance 0 after one step.
        double[] gamma = [1, 1, 1, 1];
        var pacf = new double[3];

        var complete = DurbinLevinson.TryPacf(gamma, 3, pacf, out var computed);

        Assert.False(complete);
        Assert.Equal(1, computed);
        Assert.Equal(1d, pacf[0]);

        Assert.Throws<InvalidOperationException>(() => DurbinLevinson.Pacf(gamma, 3));
    }

    [Fact]
    public void ZeroLags_IsTriviallyComplete()
    {
        double[] gamma = [2];

        Assert.True(DurbinLevinson.TryPacf(gamma, 0, [], out var computed));
        Assert.Equal(0, computed);
    }

    [Fact]
    public void TooFewAutocovariances_Throws()
    {
        double[] gamma = [1, 0.5];
        var pacf = new double[3];

        Assert.Throws<ArgumentException>(() => DurbinLevinson.TryPacf(gamma, 3, pacf, out _));
    }
}
