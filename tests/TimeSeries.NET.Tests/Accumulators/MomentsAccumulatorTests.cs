using TimeSeries.Accumulators;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Accumulators;

public class MomentsAccumulatorTests
{
    [Fact]
    public void MatchesTheMeanAndVarianceComputedInTwoPasses()
    {
        var series = SeriesGenerator.Ar1(5000, phi: 0.6, seed: 3001, sigma: 2d);

        var accumulator = new MomentsAccumulator();
        accumulator.AddRange(series);
        var actual = accumulator.Freeze();

        var mean = series.Average();
        var centred = series.Sum(value => (value - mean) * (value - mean));

        Assert.Equal(series.Length, actual.Count);
        Assert.Equal(mean, actual.Mean, tolerance: Math.Abs(mean) * 1e-12);
        Assert.Equal(centred / series.Length, actual.Variance, tolerance: 1e-10);
        Assert.Equal(centred / (series.Length - 1), actual.SampleVariance, tolerance: 1e-10);
        Assert.Equal(Math.Sqrt(centred / series.Length), actual.StandardDeviation, tolerance: 1e-10);
    }

    [Fact]
    public void ReportsTheRange()
    {
        double[] series = [4, -2, 9, 0, 3];

        var accumulator = new MomentsAccumulator();
        accumulator.AddRange(series);
        var actual = accumulator.Freeze();

        Assert.Equal(-2d, actual.Minimum);
        Assert.Equal(9d, actual.Maximum);
        Assert.Equal(2.8d, actual.Mean, tolerance: 1e-12);
    }

    [Fact]
    public void HandComputedFixture()
    {
        // mean 5; deviations -4, -2, 0, 2, 4; squares 16, 4, 0, 4, 16 summing to 40.
        double[] series = [1, 3, 5, 7, 9];

        var accumulator = new MomentsAccumulator();
        accumulator.AddRange(series);
        var actual = accumulator.Freeze();

        Assert.Equal(5d, actual.Mean, tolerance: 1e-12);
        Assert.Equal(8d, actual.Variance, tolerance: 1e-12);
        Assert.Equal(10d, actual.SampleVariance, tolerance: 1e-12);
    }

    [Fact]
    public void Add_ReadsOnlyTheCurrentObservationFromALagRow()
    {
        // The lag row carries y_t and its history; the moments want the observation only.
        // Reading more would count every value once per lag it appears in.
        var rowBased = new MomentsAccumulator();
        var scalarBased = new MomentsAccumulator();

        double[][] rows = [[5, 0, 0], [7, 5, 0], [9, 7, 5]];

        foreach (var row in rows)
        {
            rowBased.Add(row);
            scalarBased.Add(row[0]);
        }

        Assert.Equal(3d, rowBased.Freeze().Count);
        Assert.Equal(scalarBased.Freeze().Mean, rowBased.Freeze().Mean);
        Assert.Equal(7d, rowBased.Freeze().Mean, tolerance: 1e-12);
    }

    [Fact]
    public void Merge_IsExact_SoAPartitionedScanEqualsASinglePassBitForBit()
    {
        var series = new double[3000];
        var random = new Random(3002);
        for (var i = 0; i < series.Length; i++)
        {
            series[i] = random.Next(-2000, 2001);
        }

        var single = new MomentsAccumulator();
        single.AddRange(series);

        var merged = new MomentsAccumulator();
        foreach (var (start, end) in new[] { (0, 700), (700, 1900), (1900, 3000) })
        {
            var part = new MomentsAccumulator();
            part.AddRange(series.AsSpan(start, end - start));
            merged.Merge(part);
        }

        var expected = single.Freeze();
        var actual = merged.Freeze();

        Assert.Equal(expected.Count, actual.Count);
        Assert.Equal(expected.Mean, actual.Mean);
        Assert.Equal(expected.Variance, actual.Variance);
        Assert.Equal(expected.Minimum, actual.Minimum);
        Assert.Equal(expected.Maximum, actual.Maximum);
    }

    [Fact]
    public void TheOffsetKeepsTheVarianceAccurateWhenTheLevelDwarfsTheDeviations()
    {
        // A level near 10^8 with unit deviations: sum(x^2) reaches 10^16 per observation,
        // which is where a double runs out of significant digits and the variance becomes
        // the difference of two nearly equal enormous numbers.
        var series = SeriesGenerator.Gaussian(4000, seed: 3003);
        for (var i = 0; i < series.Length; i++)
        {
            series[i] += 100_000_000d;
        }

        var mean = series.Average();
        var reference = series.Sum(value => (value - mean) * (value - mean)) / series.Length;

        var unshifted = new MomentsAccumulator();
        unshifted.AddRange(series);

        var shifted = new MomentsAccumulator(offset: series[0]);
        shifted.AddRange(series);

        Assert.Equal(reference, shifted.Freeze().Variance, tolerance: reference * 1e-9);

        var shiftedError = Math.Abs(shifted.Freeze().Variance - reference);
        var unshiftedError = Math.Abs(unshifted.Freeze().Variance - reference);

        Assert.True(
            unshiftedError > shiftedError * 100d,
            $"Expected the offset to buy at least two orders of magnitude of accuracy, but " +
            $"the unshifted error was {unshiftedError:E3} against {shiftedError:E3}.");
    }

    [Theory]
    [InlineData(0d)]
    [InlineData(7d)]
    [InlineData(1234.5678d)]
    [InlineData(-98765.4321d)]
    [InlineData(1e8)]
    public void AConstantSeriesHasExactlyZeroVarianceAtEveryLevel(double constant)
    {
        // sum(x^2) - sum(x) * mean is a difference of nearly equal large numbers, and at
        // some levels the residue is around 1e-8 rather than zero — positive, so clamping
        // negatives does not catch it. The standing decision to short-circuit a constant
        // differenced series needs the variance to actually be zero, so it is read off the
        // range instead of computed.
        var series = new double[500];
        Array.Fill(series, constant);

        var accumulator = new MomentsAccumulator();
        accumulator.AddRange(series);
        var actual = accumulator.Freeze();

        Assert.Equal(0d, actual.Variance);
        Assert.Equal(0d, actual.SampleVariance);
        Assert.Equal(0d, actual.StandardDeviation);
        Assert.Equal(constant, actual.Mean);
    }

    [Fact]
    public void AnEmptyAccumulatorReportsNothingRatherThanZero()
    {
        var actual = new MomentsAccumulator().Freeze();

        Assert.Equal(0d, actual.Count);
        Assert.True(double.IsNaN(actual.Mean));
        Assert.True(double.IsNaN(actual.Variance));
    }

    [Fact]
    public void ASingleObservationHasNoSampleVariance()
    {
        var accumulator = new MomentsAccumulator();
        accumulator.Add(42d);
        var actual = accumulator.Freeze();

        Assert.Equal(42d, actual.Mean);
        Assert.Equal(0d, actual.Variance);
        Assert.True(double.IsNaN(actual.SampleVariance));
    }

    [Fact]
    public void Scale_AgesTheWeightWithoutMovingTheEstimates()
    {
        var series = SeriesGenerator.Ar1(1000, phi: 0.4, seed: 3004);

        var accumulator = new MomentsAccumulator();
        accumulator.AddRange(series);

        var before = accumulator.Freeze();
        accumulator.Scale(0.5);
        var after = accumulator.Freeze();

        Assert.Equal(before.Count * 0.5, after.Count, tolerance: 1e-12);
        Assert.Equal(before.Mean, after.Mean, tolerance: Math.Abs(before.Mean) * 1e-12);
        Assert.Equal(before.Variance, after.Variance, tolerance: before.Variance * 1e-12);
    }

    [Fact]
    public void MergingDifferentOffsets_Throws()
    {
        var target = new MomentsAccumulator(offset: 1d);

        Assert.Throws<ArgumentException>(() => target.Merge(new MomentsAccumulator(offset: 2d)));
    }

    [Fact]
    public void ANonFiniteOffset_Throws()
    {
        Assert.Throws<ArgumentException>(() => new MomentsAccumulator(double.NaN));
        Assert.Throws<ArgumentException>(() => new MomentsAccumulator(double.PositiveInfinity));
    }

    [Fact]
    public void Reset_ClearsEverythingIncludingTheRange()
    {
        var accumulator = new MomentsAccumulator();
        accumulator.AddRange([1d, 2d, 3d]);
        accumulator.Reset();
        accumulator.AddRange([10d, 20d]);

        var actual = accumulator.Freeze();

        Assert.Equal(2d, actual.Count);
        Assert.Equal(15d, actual.Mean, tolerance: 1e-12);
        Assert.Equal(10d, actual.Minimum);
        Assert.Equal(20d, actual.Maximum);
    }
}
