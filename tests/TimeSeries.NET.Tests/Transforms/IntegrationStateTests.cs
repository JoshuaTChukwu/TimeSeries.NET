using TimeSeries.Tests.TestUtils;
using TimeSeries.Transforms;

namespace TimeSeries.Tests.Transforms;

public class IntegrationStateTests
{
    [Theory]
    [MemberData(nameof(DifferenceTransformTests.Specs), MemberType = typeof(DifferenceTransformTests))]
    public void DifferenceThenIntegrate_RoundTrips(int order, int seasonalOrder, int period)
    {
        var spec = new DifferenceSpec(order, seasonalOrder, period);

        // Property-tested: the same invariant over many independently seeded series.
        for (var seed = 1; seed <= 50; seed++)
        {
            var series = SeriesGenerator.RandomWalk(300, seed, drift: 0.05, level: 20);

            var differenced = Differencing.Difference(series, spec, out var state);
            var restored = Differencing.Integrate(differenced, state);

            Assert.Equal(series.Length - spec.WarmupLength, restored.Length);

            for (var i = 0; i < restored.Length; i++)
            {
                Assert.Equal(series[spec.WarmupLength + i], restored[i], tolerance: 1e-12);
            }
        }
    }

    [Fact]
    public void TailLengthIsBoundedByTheModelOrder_NotTheSeries()
    {
        var spec = new DifferenceSpec(1, 1, 12);

        var shortState = IntegrationState.FromEnd(SeriesGenerator.Uniform(50, seed: 1), spec);
        var longState = IntegrationState.FromEnd(SeriesGenerator.Uniform(50_000, seed: 1), spec);

        Assert.Equal(spec.WarmupLength, shortState.Tail.Length);
        Assert.Equal(spec.WarmupLength, longState.Tail.Length);
        Assert.Equal(13, spec.WarmupLength);
    }

    [Theory]
    [MemberData(nameof(DifferenceTransformTests.Specs), MemberType = typeof(DifferenceTransformTests))]
    public void FromEnd_ContinuesTheSeriesPastItsLastObservation(
        int order, int seasonalOrder, int period)
    {
        var spec = new DifferenceSpec(order, seasonalOrder, period);
        var whole = SeriesGenerator.RandomWalk(280, seed: 41, drift: 0.2, level: 5);

        const int FutureLength = 24;
        var observed = whole.AsSpan(0, whole.Length - FutureLength).ToArray();
        var future = whole.AsSpan(whole.Length - FutureLength).ToArray();

        // The differenced values that belong to the future rows, as the forecaster
        // would produce them on the differenced scale.
        var differencedWhole = Differencing.Difference(whole, spec);
        var futureDifferenced = differencedWhole
            .AsSpan(differencedWhole.Length - FutureLength)
            .ToArray();

        var state = IntegrationState.FromEnd(observed, spec);
        var integrated = state.Integrate(futureDifferenced);

        Assert.Equal(FutureLength, integrated.Length);
        for (var i = 0; i < FutureLength; i++)
        {
            Assert.Equal(future[i], integrated[i], tolerance: 1e-12);
        }
    }

    [Fact]
    public void State_IsReusable_BecauseIntegrationDoesNotMutateIt()
    {
        var spec = new DifferenceSpec(2, 1, 4);
        var series = SeriesGenerator.RandomWalk(120, seed: 7, drift: 0.4);

        var differenced = Differencing.Difference(series, spec, out var state);

        var first = state.Integrate(differenced);
        var second = state.Integrate(differenced);

        Assert.Equal(first, second);
    }

    [Fact]
    public void FromStart_AndFromEnd_AgreeWhenTheSeriesIsExactlyTheWarmup()
    {
        var spec = new DifferenceSpec(1, 1, 4);
        var series = SeriesGenerator.Uniform(spec.WarmupLength, seed: 2, scale: 10);

        var fromStart = IntegrationState.FromStart(series, spec);
        var fromEnd = IntegrationState.FromEnd(series, spec);

        Assert.Equal(fromStart.Tail.ToArray(), fromEnd.Tail.ToArray());
    }

    [Fact]
    public void SeriesShorterThanWarmup_Throws()
    {
        var spec = new DifferenceSpec(1, 1, 12);
        var series = new double[12];

        var threw = false;
        try
        {
            IntegrationState.FromStart(series, spec);
        }
        catch (ArgumentException)
        {
            threw = true;
        }

        Assert.True(threw);
    }

    [Fact]
    public void ConstructedFromAWrongLengthTail_Throws()
    {
        var spec = new DifferenceSpec(1, 1, 12);
        var wrongLength = new double[12];

        var threw = false;
        try
        {
            _ = new IntegrationState(spec, wrongLength);
        }
        catch (ArgumentException)
        {
            threw = true;
        }

        Assert.True(threw);
    }

    [Fact]
    public void RoundTripThroughTheSerialisableTail_IsLossless()
    {
        var spec = new DifferenceSpec(1, 1, 4);
        var series = SeriesGenerator.RandomWalk(200, seed: 31, drift: 0.1);
        var differenced = Differencing.Difference(series, spec, out var state);

        // What M6 will write to the consumer's own database and read back.
        var wire = state.Tail.ToArray();
        var restoredState = new IntegrationState(spec, wire);

        Assert.Equal(state.Integrate(differenced), restoredState.Integrate(differenced));
    }
}
