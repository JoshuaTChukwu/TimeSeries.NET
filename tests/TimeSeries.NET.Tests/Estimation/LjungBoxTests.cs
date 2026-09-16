using TimeSeries.Statistics;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Estimation;

public class LjungBoxTests
{
    [Fact]
    public void CorrectlySpecifiedModel_RarelyRejectsWhiteNoiseResiduals()
    {
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1), PilotOrder = 8 });
        var rejections = 0;

        for (var seed = 1; seed <= 20; seed++)
        {
            var fit = model.Fit(SeriesGenerator.Arma(3_000, [0.6], [0.3], seed));

            Assert.Equal(10, fit.Diagnostics.LjungBoxLags);
            Assert.Equal(8, fit.Diagnostics.LjungBoxDegreesOfFreedom);
            Assert.InRange(fit.Diagnostics.LjungBoxPValue, 0d, 1d);

            if (fit.Diagnostics.RejectsWhiteNoiseResiduals())
            {
                rejections++;
            }
        }

        // Nominal 5% size; allow slack without letting a broken statistic through.
        Assert.True(rejections <= 4, $"Ljung-Box rejected {rejections}/20 correctly specified fits.");
    }

    [Fact]
    public void UnderSpecifiedModel_RejectsEveryTime()
    {
        // AR(1) fitted to AR(2) data with a strong second lag leaves autocorrelated residuals.
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) });

        for (var seed = 1; seed <= 10; seed++)
        {
            var fit = model.Fit(SeriesGenerator.Ar(3_000, [0.3, 0.5], seed));

            Assert.True(
                fit.Diagnostics.RejectsWhiteNoiseResiduals(0.01),
                $"Seed {seed}: p-value {fit.Diagnostics.LjungBoxPValue} did not reject an under-specified model.");
            Assert.True(fit.Diagnostics.ResidualAutocorrelations.Span[1] > 0.1, "Lag-2 residual autocorrelation should be visible.");
        }
    }

    [Fact]
    public void Disabled_LeavesTheGramAtEstimationDepth()
    {
        var series = SeriesGenerator.Arma(2_000, [0.5], [0.3], seed: 3);
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1), PilotOrder = 6, LjungBoxLags = 0 }).Fit(series);

        Assert.Equal(2_000 - 7, fit.Diagnostics.EffectiveObservations);
        Assert.Equal(0, fit.Diagnostics.LjungBoxLags);
        Assert.True(double.IsNaN(fit.Diagnostics.LjungBoxStatistic));
        Assert.True(double.IsNaN(fit.Diagnostics.LjungBoxPValue));
        Assert.False(fit.Diagnostics.RejectsWhiteNoiseResiduals());
        Assert.Empty(fit.Diagnostics.ResidualAutocorrelations.ToArray());
    }

    [Fact]
    public void TooFewLagsForTheOrder_LeavesThePValueUndefined()
    {
        // K = 2 with p + q = 2 gives zero degrees of freedom: the statistic exists, the test does not.
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1), PilotOrder = 6, LjungBoxLags = 2 })
            .Fit(SeriesGenerator.Arma(2_000, [0.5], [0.3], seed: 4));

        Assert.Equal(0, fit.Diagnostics.LjungBoxDegreesOfFreedom);
        Assert.False(double.IsNaN(fit.Diagnostics.LjungBoxStatistic));
        Assert.True(double.IsNaN(fit.Diagnostics.LjungBoxPValue));
    }

    [Fact]
    public void ConstantSeries_HasNoTest()
    {
        var fit = new ArimaModel(new ArimaOptions { Order = new(0, 1, 0) }).Fit(Enumerable.Range(0, 200).Select(t => 2d * t).ToArray());

        Assert.True(fit.Diagnostics.IsConstantSeries);
        Assert.True(double.IsNaN(fit.Diagnostics.LjungBoxPValue));
    }

    [Fact]
    public void NegativeLags_AreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ArimaModel(new ArimaOptions { LjungBoxLags = -1 }));
    }
}

public class ChiSquaredDistributionTests
{
    [Theory]
    [InlineData(3.841458820694124, 1, 0.05)]
    [InlineData(5.991464547107979, 2, 0.05)]
    [InlineData(18.307038053275146, 10, 0.05)]
    [InlineData(23.209251158954356, 10, 0.01)]
    [InlineData(2.7055434540954, 1, 0.10)]
    public void SurvivalFunction_MatchesTabulatedCriticalValues(double x, int df, double tail)
    {
        Assert.Equal(tail, ChiSquaredDistribution.SurvivalFunction(x, df), 1e-9);
    }

    [Fact]
    public void TwoDegreesOfFreedom_IsExponential()
    {
        // chi-squared(2) is exponential with mean 2: P(X > x) = exp(-x/2) exactly.
        foreach (var x in new[] { 0.1, 1d, 3d, 10d, 40d })
        {
            Assert.Equal(Math.Exp(-x / 2d), ChiSquaredDistribution.SurvivalFunction(x, 2), 1e-13);
        }
    }

    [Fact]
    public void Edges()
    {
        Assert.Equal(1d, ChiSquaredDistribution.SurvivalFunction(0d, 5));
        Assert.Equal(0d, ChiSquaredDistribution.SurvivalFunction(1e4, 5), 1e-15);
        Assert.Equal(0.5, ChiSquaredDistribution.Cdf(5.991464547107979, 2) + ChiSquaredDistribution.SurvivalFunction(5.991464547107979, 2) - 0.5, 1e-13);
        Assert.True(double.IsNaN(ChiSquaredDistribution.SurvivalFunction(double.NaN, 3)));
        Assert.Throws<ArgumentOutOfRangeException>(() => ChiSquaredDistribution.SurvivalFunction(-1d, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => ChiSquaredDistribution.SurvivalFunction(1d, 0));
    }

    [Fact]
    public void LogGamma_KnownValues()
    {
        Assert.Equal(Math.Log(Math.Sqrt(Math.PI)), ChiSquaredDistribution.LogGamma(0.5), 1e-13);
        Assert.Equal(Math.Log(24d), ChiSquaredDistribution.LogGamma(5d), 1e-13);
        Assert.Equal(0d, ChiSquaredDistribution.LogGamma(1d), 1e-14);
        Assert.Equal(0d, ChiSquaredDistribution.LogGamma(2d), 1e-14);
    }
}
