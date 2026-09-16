using TimeSeries.Statistics;

namespace TimeSeries.Tests.Statistics;

public class NormalDistributionTests
{
    [Theory]
    [InlineData(0.5, 0d)]
    [InlineData(0.975, 1.959963984540054)]
    [InlineData(0.995, 2.5758293035489004)]
    [InlineData(0.8413447460685429, 1d)]
    [InlineData(0.999, 3.090232306167813)]
    [InlineData(0.025, -1.959963984540054)]
    [InlineData(1e-10, -6.361340902404056)]
    public void Quantile_MatchesReferenceValues(double p, double expected)
    {
        Assert.Equal(expected, NormalDistribution.Quantile(p), 1e-12);
    }

    [Theory]
    [InlineData(0d, 0.5)]
    [InlineData(1d, 0.8413447460685429)]
    [InlineData(1.959963984540054, 0.975)]
    [InlineData(-2.5758293035489004, 0.005)]
    [InlineData(3d, 0.9986501019683699)]
    public void Cdf_MatchesReferenceValues(double x, double expected)
    {
        Assert.Equal(expected, NormalDistribution.Cdf(x), 1e-13);
    }

    [Fact]
    public void Cdf_AndQuantile_RoundTrip_AcrossTheWholeRange()
    {
        // Two independent algorithms (AS 241 and Marsaglia's series) agreeing to 1e-12
        // is the check on both sets of coefficients.
        foreach (var p in new[] { 1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1 - 1e-6, 1 - 1e-9 })
        {
            var z = NormalDistribution.Quantile(p);
            Assert.Equal(p, NormalDistribution.Cdf(z), 1e-12);
        }
    }

    [Fact]
    public void Cdf_FarTails_AreMonotoneAndBounded()
    {
        Assert.True(NormalDistribution.Cdf(-40) >= 0d);
        Assert.True(NormalDistribution.Cdf(40) <= 1d);
        Assert.Equal(1d, NormalDistribution.Cdf(9), 1e-15);
        Assert.Equal(0d, NormalDistribution.Cdf(-9), 1e-15);
        Assert.True(NormalDistribution.Cdf(7.5) > NormalDistribution.Cdf(7.0));
        Assert.True(NormalDistribution.Cdf(-7.5) < NormalDistribution.Cdf(-7.0));
    }

    [Fact]
    public void TwoSidedCriticalValue_Is1Point96_At95Percent()
    {
        Assert.Equal(1.959963984540054, NormalDistribution.TwoSidedCriticalValue(0.95), 1e-12);
    }

    [Theory]
    [InlineData(0d)]
    [InlineData(1d)]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    public void Quantile_OutsideOpenUnitInterval_Throws(double p)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => NormalDistribution.Quantile(p));
    }
}
