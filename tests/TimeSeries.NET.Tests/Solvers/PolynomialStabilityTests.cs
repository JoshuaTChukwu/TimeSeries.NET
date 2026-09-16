using TimeSeries.Solvers;

namespace TimeSeries.Tests.Solvers;

public class PolynomialStabilityTests
{
    [Theory]
    [InlineData(new double[] { 0.5 }, true)]
    [InlineData(new double[] { -0.9 }, true)]
    [InlineData(new double[] { 1.0 }, false)]   // unit root
    [InlineData(new double[] { 1.2 }, false)]
    [InlineData(new double[] { 0.5, 0.3 }, true)]   // roots 1.17 and -2.84
    [InlineData(new double[] { 0.5, 0.6 }, false)]  // phi1 + phi2 > 1
    [InlineData(new double[] { 1.5, -0.7 }, true)]  // complex roots, |z|^2 = 1/0.7
    public void IsStationary_AgreesWithTheRoots(double[] phi, bool expected)
    {
        Assert.Equal(expected, PolynomialStability.IsStationary(phi, out _));
    }

    [Theory]
    [InlineData(new double[] { 0.5 }, true)]
    [InlineData(new double[] { -0.5 }, true)]
    [InlineData(new double[] { 1.0 }, false)]
    [InlineData(new double[] { 2.0 }, false)]
    [InlineData(new double[] { 0.5, 0.3 }, true)]   // complex roots, |z|^2 = 1/0.3
    [InlineData(new double[] { 0.5, 2.0 }, false)]  // complex roots, |z|^2 = 1/2
    public void IsInvertible_AgreesWithTheRoots(double[] theta, bool expected)
    {
        Assert.Equal(expected, PolynomialStability.IsInvertible(theta, out _));
    }

    [Fact]
    public void EmptyPolynomial_IsTriviallyStable_WithFullMargin()
    {
        Assert.True(PolynomialStability.IsMinimumPhase([], out var margin));
        Assert.Equal(1d, margin);
    }

    [Fact]
    public void Margin_IsOneMinusTheLargestReflectionCoefficient()
    {
        PolynomialStability.IsInvertible([0.5], out var margin);
        Assert.Equal(0.5, margin, 1e-12);

        PolynomialStability.IsStationary([0.95], out margin);
        Assert.Equal(0.05, margin, 1e-12);
    }

    [Fact]
    public void Margin_IsNonPositiveOnFailure()
    {
        Assert.False(PolynomialStability.IsStationary([1.2], out var margin));
        Assert.True(margin <= 0d);
    }
}
