using TimeSeries.Statistics;

namespace TimeSeries.Tests.Statistics;

public class ErrorMetricsTests
{
    // errors: -0.5, +0.5, -0.5, +0.5
    private static readonly double[] Actual = [1, 2, 3, 4];
    private static readonly double[] Forecast = [1.5, 1.5, 3.5, 3.5];

    [Fact]
    public void Mae_HandComputed() => Assert.Equal(0.5, ErrorMetrics.Mae(Actual, Forecast), 1e-12);

    [Fact]
    public void Mse_HandComputed() => Assert.Equal(0.25, ErrorMetrics.Mse(Actual, Forecast), 1e-12);

    [Fact]
    public void Rmse_HandComputed() => Assert.Equal(0.5, ErrorMetrics.Rmse(Actual, Forecast), 1e-12);

    [Fact]
    public void Mape_HandComputed_InPercent()
    {
        // 100 * mean(0.5/1, 0.5/2, 0.5/3, 0.5/4) = 100 * (0.5 + 0.25 + 1/6 + 0.125) / 4
        var expected = 100d * (0.5 + 0.25 + (1d / 6d) + 0.125) / 4d;

        Assert.Equal(expected, ErrorMetrics.Mape(Actual, Forecast), 1e-12);
        Assert.Equal(26.041667, ErrorMetrics.Mape(Actual, Forecast), 1e-6);
    }

    [Fact]
    public void Smape_HandComputed_InPercent()
    {
        // 200 * mean(0.5/2.5, 0.5/3.5, 0.5/6.5, 0.5/7.5)
        var expected = 200d * ((0.5 / 2.5) + (0.5 / 3.5) + (0.5 / 6.5) + (0.5 / 7.5)) / 4d;

        Assert.Equal(expected, ErrorMetrics.Smape(Actual, Forecast), 1e-12);
        Assert.Equal(24.322344, ErrorMetrics.Smape(Actual, Forecast), 1e-6);
    }

    [Fact]
    public void Smape_WhereBothAreZero_ContributesZeroRatherThanNaN()
    {
        double[] actual = [0, 2];
        double[] forecast = [0, 2];

        Assert.Equal(0d, ErrorMetrics.Smape(actual, forecast));
    }

    [Fact]
    public void Mape_WithAZeroActual_Throws_NamingTheIndex()
    {
        // The standing decision: never a silently dropped point.
        double[] actual = [1, 0, 3];
        double[] forecast = [1, 1, 3];

        var exception = Assert.Throws<ArgumentException>(() => ErrorMetrics.Mape(actual, forecast));

        Assert.Contains("actual[1]", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Mase_HandComputed()
    {
        // Naive MAE on training 1..5 is 1; forecast MAE is 0.5.
        double[] training = [1, 2, 3, 4, 5];
        double[] actual = [6, 7];
        double[] forecast = [6.5, 6.5];

        Assert.Equal(0.5, ErrorMetrics.Mase(actual, forecast, training), 1e-12);
    }

    [Fact]
    public void Mase_WithSeasonalPeriod_HandComputed()
    {
        // Period 2: |3-1| + |4-2| + |5-3| + |6-4| = 8 over 4 terms  =>  scale 2.
        double[] training = [1, 2, 3, 4, 5, 6];
        double[] actual = [7, 8];
        double[] forecast = [8, 9];

        Assert.Equal(0.5, ErrorMetrics.Mase(actual, forecast, training, period: 2), 1e-12);
    }

    [Fact]
    public void Mase_WithAnExactNaiveForecast_Throws()
    {
        double[] training = [1, 2, 1, 2, 1, 2];
        double[] actual = [1];
        double[] forecast = [1];

        Assert.Throws<ArgumentException>(() => ErrorMetrics.Mase(actual, forecast, training, period: 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => ErrorMetrics.Mase(actual, forecast, training, period: 0));
        Assert.Throws<ArgumentException>(() => ErrorMetrics.Mase(actual, forecast, [1, 2], period: 2));
    }

    [Fact]
    public void Coverage_HandComputed()
    {
        double[] actual = [1, 2, 3, 4];
        double[] lower = [0, 2.5, 2, 3];
        double[] upper = [2, 3, 4, 5];

        Assert.Equal(0.75, ErrorMetrics.Coverage(actual, lower, upper), 1e-12);
    }

    [Fact]
    public void MismatchedLengths_Throw()
    {
        double[] actual = [1, 2, 3];
        double[] forecast = [1, 2];

        Assert.Throws<ArgumentException>(() => ErrorMetrics.Mae(actual, forecast));
        Assert.Throws<ArgumentException>(() => ErrorMetrics.Mse(actual, forecast));
        Assert.Throws<ArgumentException>(() => ErrorMetrics.Mape(actual, forecast));
        Assert.Throws<ArgumentException>(() => ErrorMetrics.Smape(actual, forecast));
        Assert.Throws<ArgumentException>(() => ErrorMetrics.Coverage(actual, forecast, actual));
    }

    [Fact]
    public void EmptyInputs_Throw()
    {
        Assert.Throws<ArgumentException>(() => ErrorMetrics.Mae([], []));
    }
}
