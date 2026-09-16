namespace TimeSeries.Tests.TestUtils;
public static class TestData
{
    /// <summary>
    /// Basic increasing trend (e.g., 10, 13, 17, 21)
    /// </summary>
    public static double[] TrendingSeries => new double[] { 10, 13, 17, 21 };

    /// <summary>
    /// Series with constant first difference (linear growth)
    /// </summary>
    public static double[] LinearSeries => new double[] { 5, 10, 15, 20 };

    /// <summary>
    /// Series with changing differences
    /// </summary>
    public static double[] NonLinearSeries => new double[] { 2, 5, 9, 14 };

    /// <summary>
    /// Short series for invalid test cases
    /// </summary>
    public static double[] ShortSeries => new double[] { 3, 7 };

    /// <summary>
    /// Flat series (no change)
    /// </summary>
    public static double[] FlatSeries => new double[] { 10, 10, 10, 10 };
}