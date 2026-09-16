using TimeSeries.Transforms;

namespace TimeSeries.Tests.Transforms;

public class DifferenceSpecTests
{
    [Fact]
    public void None_IsIdentity_WithNoWarmup()
    {
        Assert.True(DifferenceSpec.None.IsIdentity);
        Assert.Equal(0, DifferenceSpec.None.WarmupLength);
    }

    [Theory]
    [InlineData(0, 0, 0, 0)]
    [InlineData(1, 0, 0, 1)]
    [InlineData(2, 0, 0, 2)]
    [InlineData(0, 1, 12, 12)]
    [InlineData(1, 1, 12, 13)]
    [InlineData(2, 1, 4, 6)]
    [InlineData(1, 2, 7, 15)]
    public void WarmupLength_IsOrderPlusSeasonalOrderTimesPeriod(
        int order, int seasonalOrder, int period, int expected)
    {
        var spec = new DifferenceSpec(order, seasonalOrder, period);

        Assert.Equal(expected, spec.WarmupLength);
    }

    [Fact]
    public void NegativeOrder_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DifferenceSpec(-1));
    }

    [Fact]
    public void NegativeSeasonalOrder_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DifferenceSpec(1, -1, 12));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void SeasonalDifferenceWithPeriodBelowTwo_Throws(int period)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DifferenceSpec(0, 1, period));
    }

    [Fact]
    public void PeriodIsNormalisedAwayWhenThereIsNoSeasonalDifference()
    {
        // Same transform, so the two must compare equal.
        Assert.Equal(new DifferenceSpec(1), new DifferenceSpec(1, 0, 12));
        Assert.Equal(0, new DifferenceSpec(1, 0, 12).Period);
    }

    [Fact]
    public void ToString_ShowsThePolynomial()
    {
        Assert.Equal("identity", DifferenceSpec.None.ToString());
        Assert.Equal("(1-B)^1", new DifferenceSpec(1).ToString());
        Assert.Equal("(1-B^12)^1", new DifferenceSpec(0, 1, 12).ToString());
        Assert.Equal("(1-B)^1(1-B^12)^1", new DifferenceSpec(1, 1, 12).ToString());
    }
}
