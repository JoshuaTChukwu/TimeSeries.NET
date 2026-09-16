using TimeSeries.Accumulators;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Accumulators;

public class KpssAccumulatorTests
{
    /// <summary>The textbook two-pass computation, for the one-pass algebra to be checked against.</summary>
    private static double Naive(double[] y, int bandwidth)
    {
        var n = y.Length;
        var mean = y.Average();
        var partial = 0d;
        var sumSquared = 0d;

        foreach (var value in y)
        {
            partial += value - mean;
            sumSquared += partial * partial;
        }

        double Gamma(int k)
        {
            var sum = 0d;
            for (var t = k; t < n; t++)
            {
                sum += (y[t] - mean) * (y[t - k] - mean);
            }

            return sum / n;
        }

        var longRun = Gamma(0);
        for (var k = 1; k <= bandwidth; k++)
        {
            longRun += 2d * (1d - (k / (bandwidth + 1d))) * Gamma(k);
        }

        return sumSquared / (n * n * longRun);
    }

    private static KpssResult Run(double[] y, int maxLag = 32)
    {
        var accumulator = new KpssAccumulator(maxLag);
        var window = new LagWindow(maxLag);
        var row = new double[maxLag + 1];

        foreach (var value in y)
        {
            window.Push(value);
            var available = window.CopyRow(row);
            accumulator.Add(row.AsSpan(0, available));
        }

        return accumulator.Freeze();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void OnePassStatistic_EqualsTheTwoPassComputation(int seed)
    {
        var y = SeriesGenerator.Ar1(1_500, 0.6, seed);
        var result = Run(y);

        Assert.Equal(Naive(y, result.Bandwidth), result.Statistic, 1e-9);
        Assert.Equal((int)Math.Floor(4d * Math.Pow(15d, 0.25)), result.Bandwidth);
    }

    [Fact]
    public void StationarySeries_IsNotRejected()
    {
        var rejections = 0;
        for (var seed = 1; seed <= 20; seed++)
        {
            if (Run(SeriesGenerator.Ar1(2_000, 0.5, seed)).RejectsStationarity())
            {
                rejections++;
            }
        }

        // 5% nominal size; allow generous slack without letting a broken test pass.
        Assert.True(rejections <= 4, $"KPSS rejected stationarity for {rejections}/20 stationary series.");
    }

    [Fact]
    public void RandomWalk_IsRejected()
    {
        for (var seed = 1; seed <= 20; seed++)
        {
            var result = Run(SeriesGenerator.RandomWalk(2_000, seed));
            Assert.True(result.RejectsStationarity(0.01), $"Seed {seed}: statistic {result.Statistic} did not reject a random walk.");
        }
    }

    [Fact]
    public void LevelShift_DoesNotChangeTheStatistic()
    {
        var y = SeriesGenerator.Ar1(1_000, 0.4, seed: 9);
        var lifted = y.Select(v => v + 25_000d).ToArray();

        Assert.Equal(Run(y).Statistic, Run(lifted).Statistic, 1e-6);
    }

    [Fact]
    public void ConstantSeries_IsTriviallyStationary()
    {
        var result = Run(Enumerable.Repeat(3.5, 200).ToArray());

        Assert.Equal(0d, result.Statistic);
        Assert.False(result.RejectsStationarity());
    }

    [Fact]
    public void TooFewObservations_GiveNaN()
    {
        var result = Run([1d, 2d, 3d]);
        Assert.True(double.IsNaN(result.Statistic));
        Assert.False(result.RejectsStationarity());
    }

    [Fact]
    public void CriticalValues_AndMerge()
    {
        Assert.Equal(0.463, KpssResult.CriticalValue(0.05));
        Assert.Equal(0.739, KpssResult.CriticalValue(0.01));
        Assert.Throws<ArgumentOutOfRangeException>(() => KpssResult.CriticalValue(0.07));

        var a = new KpssAccumulator();
        var b = new KpssAccumulator();
        Assert.Throws<NotSupportedException>(() => a.Merge(b));
    }
}
