using TimeSeries.Forecasting;
using TimeSeries.Transforms;

namespace TimeSeries.Tests.Forecasting;

public class PsiWeightsTests
{
    [Fact]
    public void Ar1_PsiIsPhiToTheJ()
    {
        var psi = PsiWeights.Arma([0.6], [], 6);

        for (var j = 0; j < 6; j++)
        {
            Assert.Equal(Math.Pow(0.6, j), psi[j], 1e-14);
        }
    }

    [Fact]
    public void Ma1_PsiIsOneThetaThenZero()
    {
        Assert.Equal([1d, 0.4, 0d, 0d], PsiWeights.Arma([], [0.4], 4));
    }

    [Fact]
    public void Arma11_ClosedForm()
    {
        // psi_j = (phi + theta) phi^(j-1) for j >= 1
        const double Phi = 0.5;
        const double Theta = 0.3;
        var psi = PsiWeights.Arma([Phi], [Theta], 8);

        Assert.Equal(1d, psi[0]);
        for (var j = 1; j < 8; j++)
        {
            Assert.Equal((Phi + Theta) * Math.Pow(Phi, j - 1), psi[j], 1e-14);
        }
    }

    [Fact]
    public void Integrate_OneDifference_CumulatesOnce()
    {
        var psi = new double[] { 1, 0, 0, 0, 0 };

        Assert.Equal([1d, 1d, 1d, 1d, 1d], PsiWeights.Integrate(psi, new DifferenceSpec(1)));
    }

    [Fact]
    public void Integrate_TwoDifferences_GivesTriangularWeights()
    {
        var psi = new double[] { 1, 0, 0, 0, 0 };

        Assert.Equal([1d, 2d, 3d, 4d, 5d], PsiWeights.Integrate(psi, new DifferenceSpec(2)));
    }

    [Fact]
    public void Integrate_SeasonalDifference_RepeatsEveryPeriod()
    {
        var psi = new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0 };

        Assert.Equal([1d, 0d, 0d, 0d, 1d, 0d, 0d, 0d, 1d], PsiWeights.Integrate(psi, new DifferenceSpec(0, 1, 4)));
    }

    [Fact]
    public void StandardErrors_AreSigmaRootCumulativeSquares()
    {
        var se = PsiWeights.StandardErrors([1d, 1d, 1d, 1d], 4d);

        for (var h = 1; h <= 4; h++)
        {
            Assert.Equal(2d * Math.Sqrt(h), se[h - 1], 1e-14);
        }
    }

    [Fact]
    public void ZeroCount_IsEmpty()
    {
        Assert.Empty(PsiWeights.Arma([0.5], [0.3], 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => PsiWeights.Arma([], [], -1));
    }
}

public class ForecastHorizonTests
{
    [Fact]
    public void Periods_NeedNoFrequency()
    {
        Assert.Equal(10, ForecastHorizon.Periods(10).Resolve(null));
        Assert.Equal(10, ForecastHorizon.Periods(10).Resolve(SeriesFrequency.Monthly));
    }

    [Theory]
    [InlineData(SeriesFrequency.Monthly, 24)]
    [InlineData(SeriesFrequency.Quarterly, 8)]
    [InlineData(SeriesFrequency.Yearly, 2)]
    [InlineData(SeriesFrequency.Weekly, 104)]
    [InlineData(SeriesFrequency.BusinessDaily, 504)]
    [InlineData(SeriesFrequency.Daily, 730)]
    public void TwoYears_ResolvesAgainstTheFrequency(SeriesFrequency frequency, int expected)
    {
        Assert.Equal(expected, ForecastHorizon.Years(2).Resolve(frequency));
    }

    [Fact]
    public void MonthsAndDays_Resolve()
    {
        Assert.Equal(2, ForecastHorizon.Months(6).Resolve(SeriesFrequency.Quarterly));
        Assert.Equal(6, ForecastHorizon.Months(6).Resolve(SeriesFrequency.Monthly));
        Assert.Equal(126, ForecastHorizon.Years(0.5).Resolve(SeriesFrequency.BusinessDaily));
        Assert.Equal(30, ForecastHorizon.Days(30).Resolve(SeriesFrequency.Daily));
        Assert.Equal(720, ForecastHorizon.Days(30).Resolve(SeriesFrequency.Hourly));
    }

    [Fact]
    public void CalendarHorizon_WithoutFrequency_Throws()
    {
        var exception = Assert.Throws<ForecastHorizonException>(() => ForecastHorizon.Years(2).Resolve(null));

        Assert.Contains("SeriesFrequency", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void HorizonRoundingToZero_Throws()
    {
        Assert.Throws<ForecastHorizonException>(() => ForecastHorizon.Days(3).Resolve(SeriesFrequency.Monthly));
    }

    [Fact]
    public void NonPositiveAmounts_AreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ForecastHorizon.Periods(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ForecastHorizon.Days(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ForecastHorizon.Months(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ForecastHorizon.Years(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ForecastHorizon.Years(double.NaN));
    }

    [Fact]
    public void ToString_ReadsNaturally()
    {
        Assert.Equal("2 years", ForecastHorizon.Years(2).ToString());
        Assert.Equal("12 periods", ForecastHorizon.Periods(12).ToString());
    }
}
