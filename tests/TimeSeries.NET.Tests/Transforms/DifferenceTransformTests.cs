using TimeSeries.Tests.TestUtils;
using TimeSeries.Transforms;

namespace TimeSeries.Tests.Transforms;

public class DifferenceTransformTests
{
    public static TheoryData<int, int, int> Specs => new()
    {
        { 0, 0, 0 },
        { 1, 0, 0 },
        { 2, 0, 0 },
        { 3, 0, 0 },
        { 0, 1, 4 },
        { 0, 1, 12 },
        { 0, 2, 4 },
        { 1, 1, 4 },
        { 1, 1, 12 },
        { 2, 1, 7 },
        { 1, 2, 4 },
    };

    [Theory]
    [MemberData(nameof(Specs))]
    public void Transform_MatchesNaiveRepeatedDifferencing(int order, int seasonalOrder, int period)
    {
        var series = SeriesGenerator.RandomWalk(240, seed: 17, drift: 0.3);
        var spec = new DifferenceSpec(order, seasonalOrder, period);

        var expected = Naive.Difference(series, order, seasonalOrder, period);
        var actual = Differencing.Difference(series, spec);

        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], precision: 10);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(13)]
    [InlineData(1024)]
    public void BatchBoundariesDoNotChangeTheResult(int batchSize)
    {
        var series = SeriesGenerator.RandomWalk(500, seed: 23, drift: -0.1);
        var spec = new DifferenceSpec(1, 1, 12);

        var oneShot = Differencing.Difference(series, spec);

        var transform = new DifferenceTransform(spec);
        var scratch = new double[batchSize];
        var batched = new List<double>();

        for (var start = 0; start < series.Length; start += batchSize)
        {
            var take = Math.Min(batchSize, series.Length - start);
            var produced = transform.Transform(series.AsSpan(start, take), scratch);
            batched.AddRange(scratch.AsSpan(0, produced).ToArray());
        }

        // Bit-for-bit: a batch boundary is not allowed to be visible in the output.
        Assert.Equal(oneShot, batched);
    }

    [Fact]
    public void ProducesNothingUntilWarmupIsConsumed()
    {
        var spec = new DifferenceSpec(1, 1, 12);
        var transform = new DifferenceTransform(spec);
        var scratch = new double[64];

        Assert.Equal(13, transform.WarmupLength);

        var series = SeriesGenerator.Uniform(13, seed: 5);
        Assert.Equal(0, transform.Transform(series, scratch));
        Assert.True(transform.IsWarm);

        Assert.Equal(1, transform.Transform([1.0], scratch));
    }

    [Fact]
    public void Reset_ReturnsTheTransformToItsWarmupState()
    {
        var spec = new DifferenceSpec(2);
        var transform = new DifferenceTransform(spec);
        var scratch = new double[16];
        var series = SeriesGenerator.Uniform(10, seed: 9);

        var first = new double[8];
        var count = transform.Transform(series, scratch);
        scratch.AsSpan(0, count).CopyTo(first);

        transform.Reset();
        Assert.False(transform.IsWarm);

        var second = new double[8];
        count = transform.Transform(series, scratch);
        scratch.AsSpan(0, count).CopyTo(second);

        Assert.Equal(first, second);
    }

    [Fact]
    public void CaptureState_BeforeWarmup_Throws()
    {
        var transform = new DifferenceTransform(new DifferenceSpec(1, 1, 12));
        var scratch = new double[8];

        transform.Transform(SeriesGenerator.Uniform(8, seed: 3), scratch);

        Assert.Throws<InvalidOperationException>(() => transform.CaptureState());
    }

    [Fact]
    public void Transform_WithUndersizedOutput_Throws()
    {
        var transform = new DifferenceTransform(new DifferenceSpec(1));
        var input = new double[10];
        var tooSmall = new double[9];

        // Span arguments cannot cross a lambda boundary, so the call is made directly.
        var threw = false;
        try
        {
            transform.Transform(input, tooSmall);
        }
        catch (ArgumentException)
        {
            threw = true;
        }

        Assert.True(threw);
    }

    [Fact]
    public void IdentitySpec_PassesTheSeriesThroughUnchanged()
    {
        var series = SeriesGenerator.Uniform(50, seed: 11, scale: 100);

        var result = Differencing.Difference(series, DifferenceSpec.None);

        Assert.Equal(series, result);
    }
}

public class DifferenceTransformContinuationTests
{
    [Theory]
    [MemberData(nameof(DifferenceTransformTests.Specs), MemberType = typeof(DifferenceTransformTests))]
    public void Continue_PicksUpExactlyWhereTheStateWasCaptured(int order, int seasonalOrder, int period)
    {
        var spec = new DifferenceSpec(order, seasonalOrder, period);
        var series = SeriesGenerator.RandomWalk(300, seed: 61, drift: 0.2);
        const int Split = 180;

        var oneShot = Differencing.Difference(series, spec);

        var state = IntegrationState.FromEnd(series.AsSpan(0, Split), spec);
        var continued = DifferenceTransform.Continue(state);
        Assert.True(continued.IsWarm);

        var rest = new double[series.Length - Split];
        var produced = continued.Transform(series.AsSpan(Split), rest);

        // Every row after the split produces one output, and the outputs are bit-for-bit
        // those of the single traversal.
        Assert.Equal(series.Length - Split, produced);
        Assert.Equal(oneShot.Skip(oneShot.Length - produced).ToArray(), rest);
    }

    [Fact]
    public void Continue_RejectsNull()
    {
        Assert.Throws<ArgumentNullException>(() => DifferenceTransform.Continue(null!));
    }
}
