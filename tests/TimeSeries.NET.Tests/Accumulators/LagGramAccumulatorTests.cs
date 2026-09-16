using TimeSeries.Accumulators;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Accumulators;

public class LagGramAccumulatorTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(12)]
    public void MatchesTheGramMatrixFormedTheLongWayRound(int lagDepth)
    {
        var series = SeriesGenerator.Ar1(400, phi: 0.6, seed: 101);

        var actual = new AccumulatorScan(lagDepth).Scan(series).Gram.Freeze();
        var expected = Naive.Gram(series, lagDepth);
        var expectedSums = Naive.GramSums(series, lagDepth);

        Assert.Equal(lagDepth + 1, actual.Dimension);
        Assert.Equal(series.Length - lagDepth, actual.Count);
        Assert.Equal(series.Length - lagDepth, actual.Rows);

        for (var i = 0; i <= lagDepth; i++)
        {
            Assert.Equal(expectedSums[i], actual.Sums.Span[i], tolerance: 1e-9);

            for (var j = 0; j <= lagDepth; j++)
            {
                Assert.Equal(expected[i, j], actual[i, j], tolerance: 1e-9);
            }
        }
    }

    [Fact]
    public void TheFrozenMatrixIsSymmetric()
    {
        var series = SeriesGenerator.Ar1(300, phi: -0.4, seed: 202);
        var gram = new AccumulatorScan(lagDepth: 6).Scan(series).Gram.Freeze();

        for (var i = 0; i < gram.Dimension; i++)
        {
            for (var j = 0; j < gram.Dimension; j++)
            {
                Assert.Equal(gram[i, j], gram[j, i]);
            }
        }
    }

    [Fact]
    public void TheGramIsNotToeplitz_WhichIsWhyAutocovariancesAreAccumulatedSeparately()
    {
        var series = SeriesGenerator.Ar1(500, phi: 0.75, seed: 303);
        var gram = new AccumulatorScan(lagDepth: 4).Scan(series).Gram.Freeze();

        // Same lag distance, row ranges offset by one, therefore different sums. If this
        // ever became an equality, deriving the ACF from G would start to look safe and
        // Durbin-Levinson would lose the positive semi-definiteness it depends on.
        Assert.NotEqual(gram[0, 1], gram[1, 2]);
        Assert.NotEqual(gram[0, 2], gram[2, 4]);
    }

    [Fact]
    public void OffsetShiftsTheMatrixExactlyAsShiftingTheSeriesWould()
    {
        var series = SeriesGenerator.Ar1(200, phi: 0.5, seed: 404, sigma: 1d);
        for (var i = 0; i < series.Length; i++)
        {
            series[i] += 100_000d;
        }

        var offset = series[0];
        var gram = new AccumulatorScan(lagDepth: 3, offset).Scan(series).Gram.Freeze();
        var expected = Naive.Gram(series, lagDepth: 3, offset);

        Assert.Equal(offset, gram.Offset);

        for (var i = 0; i < gram.Dimension; i++)
        {
            for (var j = 0; j < gram.Dimension; j++)
            {
                Assert.Equal(expected[i, j], gram[i, j], tolerance: 1e-9);
            }
        }
    }

    [Fact]
    public void PartitionedScanWithWarmup_EqualsASinglePass_BitForBit()
    {
        // Whole numbers, so every product and every partial sum is exactly representable.
        // That strips floating-point associativity out of the question and leaves the
        // merge logic itself — no double-counted warm-up row, no dropped boundary row —
        // as the only thing the assertion can be measuring.
        var series = new double[2000];
        var random = new Random(505);
        for (var i = 0; i < series.Length; i++)
        {
            series[i] = random.Next(-1000, 1001);
        }

        const int LagDepth = 5;
        var single = new AccumulatorScan(LagDepth).Scan(series).Gram.Freeze();

        var partitioned = new AccumulatorScan(LagDepth).ScanRange(series, 0, 640);
        foreach (var (start, end) in new[] { (640, 1280), (1280, 2000) })
        {
            partitioned.Merge(new AccumulatorScan(LagDepth).ScanRange(series, start, end));
        }

        var merged = partitioned.Gram.Freeze();

        Assert.Equal(single.Count, merged.Count);
        Assert.Equal(single.Rows, merged.Rows);
        Assert.Equal(single.Values.ToArray(), merged.Values.ToArray());
        Assert.Equal(single.Sums.ToArray(), merged.Sums.ToArray());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]
    [InlineData(64)]
    public void PartitionedScanWithWarmup_EqualsASinglePass_OnRealValuedData(int partitions)
    {
        var series = SeriesGenerator.Ar1(4000, phi: 0.8, seed: 606);
        const int LagDepth = 9;

        var single = new AccumulatorScan(LagDepth).Scan(series).Gram.Freeze();

        var size = (series.Length + partitions - 1) / partitions;
        AccumulatorScan? merged = null;

        for (var start = 0; start < series.Length; start += size)
        {
            var end = Math.Min(series.Length, start + size);
            var part = new AccumulatorScan(LagDepth).ScanRange(series, start, end);

            if (merged is null)
            {
                merged = part;
            }
            else
            {
                merged.Merge(part);
            }
        }

        var actual = merged!.Gram.Freeze();

        Assert.Equal(single.Count, actual.Count);
        Assert.Equal(single.Rows, actual.Rows);

        for (var i = 0; i < single.Dimension; i++)
        {
            for (var j = 0; j < single.Dimension; j++)
            {
                Assert.Equal(single[i, j], actual[i, j], tolerance: Math.Abs(single[i, j]) * 1e-12);
            }
        }
    }

    [Fact]
    public void Freeze_IsPureAndRepeatable()
    {
        var series = SeriesGenerator.Ar1(150, phi: 0.3, seed: 707);
        var accumulator = new AccumulatorScan(lagDepth: 4).Scan(series).Gram;

        var first = accumulator.Freeze();
        var second = accumulator.Freeze();

        Assert.Equal(first.Values.ToArray(), second.Values.ToArray());
        Assert.Equal(first.Count, second.Count);

        // Distinct backing arrays, so a caller cannot reach through one result and
        // disturb the other or the accumulator behind them. ReadOnlyMemory equality
        // compares the underlying object, offset and length, so this is that check.
        Assert.False(first.Values.Equals(second.Values));
    }

    [Fact]
    public void Scale_AgesTheWholeStateByTheForgettingFactor()
    {
        var series = SeriesGenerator.Ar1(200, phi: 0.5, seed: 808);
        var accumulator = new AccumulatorScan(lagDepth: 3).Scan(series).Gram;

        var before = accumulator.Freeze();
        accumulator.Scale(0.5);
        var after = accumulator.Freeze();

        Assert.Equal(before.Count * 0.5, after.Count, tolerance: 1e-12);

        for (var i = 0; i < before.Dimension; i++)
        {
            Assert.Equal(before.Sums.Span[i] * 0.5, after.Sums.Span[i], tolerance: 1e-9);

            for (var j = 0; j < before.Dimension; j++)
            {
                Assert.Equal(before[i, j] * 0.5, after[i, j], tolerance: Math.Abs(before[i, j]) * 1e-12);
            }
        }

        // The audit count is deliberately untouched: it records rows read, not weight.
        Assert.Equal(before.Rows, after.Rows);
    }

    [Theory]
    [InlineData(0d)]
    [InlineData(-0.5d)]
    [InlineData(1.5d)]
    public void Scale_OutsideTheUnitInterval_Throws(double lambda)
    {
        var accumulator = new LagGramAccumulator(lagDepth: 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => accumulator.Scale(lambda));
    }

    [Fact]
    public void AddingAnIncompleteRow_Throws()
    {
        var accumulator = new LagGramAccumulator(lagDepth: 3);
        var shortRow = new double[3];

        var threw = false;
        try
        {
            accumulator.Add(shortRow);
        }
        catch (ArgumentException)
        {
            threw = true;
        }

        Assert.True(threw);
    }

    [Fact]
    public void MergingMismatchedState_Throws()
    {
        var target = new LagGramAccumulator(lagDepth: 3, offset: 10d);

        Assert.Throws<ArgumentException>(() => target.Merge(new LagGramAccumulator(lagDepth: 4, offset: 10d)));
        Assert.Throws<ArgumentException>(() => target.Merge(new LagGramAccumulator(lagDepth: 3, offset: 11d)));
    }

    [Fact]
    public void CompensatedSummationAgreesWithPlainSummationOnWellConditionedData()
    {
        var series = SeriesGenerator.Ar1(2000, phi: 0.4, seed: 909);

        var plain = new AccumulatorScan(lagDepth: 5).Scan(series).Gram.Freeze();
        var compensated = new AccumulatorScan(lagDepth: 5, compensated: true).Scan(series).Gram.Freeze();

        for (var i = 0; i < plain.Dimension; i++)
        {
            for (var j = 0; j < plain.Dimension; j++)
            {
                Assert.Equal(plain[i, j], compensated[i, j], tolerance: Math.Abs(plain[i, j]) * 1e-12);
            }
        }
    }

    [Fact]
    public void CompensatedPartitionedScan_AlsoMergesToTheSinglePassResult()
    {
        var series = SeriesGenerator.Ar1(3000, phi: 0.7, seed: 1010);
        const int LagDepth = 6;

        var single = new AccumulatorScan(LagDepth, compensated: true).Scan(series).Gram.Freeze();

        var merged = new AccumulatorScan(LagDepth, compensated: true).ScanRange(series, 0, 1000);
        merged.Merge(new AccumulatorScan(LagDepth, compensated: true).ScanRange(series, 1000, 2000));
        merged.Merge(new AccumulatorScan(LagDepth, compensated: true).ScanRange(series, 2000, 3000));

        var actual = merged.Gram.Freeze();

        for (var i = 0; i < single.Dimension; i++)
        {
            for (var j = 0; j < single.Dimension; j++)
            {
                Assert.Equal(single[i, j], actual[i, j], tolerance: Math.Abs(single[i, j]) * 1e-13);
            }
        }
    }
}
