using TimeSeries.Statistics;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Statistics;

public class AcfPacfTests
{
    [Fact]
    public void Acf_OfSimulatedAr1_IsPhiToTheK()
    {
        const double Phi = 0.7;
        var series = SeriesGenerator.Ar1(20_000, Phi, seed: 77);

        var acf = Acf.Compute(series, 6).Acf.Span;

        Assert.Equal(1d, acf[0]);
        for (var k = 1; k <= 6; k++)
        {
            Assert.Equal(Math.Pow(Phi, k), acf[k], 0.03);
        }
    }

    [Fact]
    public void Acf_IsInvariantToTheSeriesLevel()
    {
        // The offset is what keeps a level of 40 000 from swamping deviations of 1.
        var centred = SeriesGenerator.Ar1(5_000, 0.6, seed: 3);
        var shifted = centred.Select(v => v + 40_000d).ToArray();

        var a = Acf.Compute(centred, 5).Acf.Span;
        var b = Acf.Compute(shifted, 5).Acf.Span;

        for (var k = 0; k <= 5; k++)
        {
            Assert.Equal(a[k], b[k], 1e-9);
        }
    }

    [Fact]
    public void WhiteNoiseBound_At95PercentAnd100Observations()
    {
        Assert.Equal(0.1959963984540054, Acf.WhiteNoiseBound(100), 1e-12);
    }

    [Fact]
    public void Acf_OfWhiteNoise_StaysInsideTheBound()
    {
        const int N = 5_000;
        var noise = SeriesGenerator.Gaussian(N, seed: 11);
        var acf = Acf.Compute(noise, 20).Acf.Span;

        // 99.9% band so the test does not flake; 20 lags of true white noise.
        var bound = Acf.WhiteNoiseBound(N, confidence: 0.999);

        for (var k = 1; k <= 20; k++)
        {
            Assert.True(Math.Abs(acf[k]) < bound, $"White-noise ACF at lag {k} is {acf[k]}, beyond ±{bound}.");
        }
    }

    [Fact]
    public void Acf_WithNaN_ThrowsNamingTheIndex()
    {
        double[] series = [1, 2, double.NaN, 4, 5];

        var exception = Assert.Throws<InvalidSeriesException>(() => Acf.Compute(series, 2));

        Assert.Equal(2, exception.Index);
        Assert.True(double.IsNaN(exception.Value));
    }

    [Fact]
    public void Acf_WithTooFewObservations_Throws()
    {
        double[] series = [1, 2, 3];

        Assert.Throws<ArgumentException>(() => Acf.Compute(series, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => Acf.Compute(series, -1));
    }

    [Fact]
    public void Pacf_OfAConstantSeries_IsAllZero()
    {
        var pacf = Pacf.Compute(TestData.FlatSeries, 2);

        Assert.Equal([0d, 0d], pacf);
    }

    [Fact]
    public void Pacf_OfSimulatedAr1_IsPhiThenNoise()
    {
        const double Phi = 0.8;
        const int N = 20_000;
        var series = SeriesGenerator.Ar1(N, Phi, seed: 91);

        var pacf = Pacf.Compute(series, 8);
        var bound = 4d / Math.Sqrt(N);

        Assert.Equal(Phi, pacf[0], 0.03);
        for (var k = 1; k < 8; k++)
        {
            Assert.True(Math.Abs(pacf[k]) < bound, $"PACF at lag {k + 1} is {pacf[k]}, beyond ±{bound}.");
        }
    }

    [Fact]
    public void Pacf_FromAutocovariances_RefusesLagsBeyondTheEstimate()
    {
        var autocovariances = Acf.Compute(SeriesGenerator.Uniform(50, seed: 1), 5);

        Assert.Throws<ArgumentOutOfRangeException>(() => Pacf.FromAutocovariances(autocovariances, 6));
        Assert.Throws<ArgumentNullException>(() => Pacf.FromAutocovariances(null!, 1));
    }

    [Fact]
    public void Pacf_FromAutocovariances_AgreesWithCompute()
    {
        var series = SeriesGenerator.Ar(2_000, [0.4, 0.2], seed: 5);
        var autocovariances = Acf.Compute(series, 6);

        Assert.Equal(Pacf.Compute(series, 6), Pacf.FromAutocovariances(autocovariances, 6));
    }
}
