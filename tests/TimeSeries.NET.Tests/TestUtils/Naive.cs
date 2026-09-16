namespace TimeSeries.Tests.TestUtils;

/// <summary>
/// Deliberately naive reference implementations, written straight from the definition
/// with no shared code with the library. Their job is to disagree with the library when
/// the library is wrong, so they trade every ounce of efficiency for being obviously
/// correct by inspection.
/// </summary>
public static class Naive
{
    /// <summary>Repeated whole-array differencing: D seasonal passes, then d non-seasonal.</summary>
    public static double[] Difference(double[] series, int order, int seasonalOrder, int period)
    {
        var current = series;

        for (var i = 0; i < seasonalOrder; i++)
        {
            current = Lag(current, period);
        }

        for (var i = 0; i < order; i++)
        {
            current = Lag(current, 1);
        }

        return current;
    }

    /// <summary>One <c>(1 - B^k)</c> pass over a whole array.</summary>
    public static double[] Lag(double[] series, int lag)
    {
        if (series.Length <= lag)
        {
            return [];
        }

        var result = new double[series.Length - lag];

        for (var i = lag; i < series.Length; i++)
        {
            result[i - lag] = series[i] - series[i - lag];
        }

        return result;
    }

    /// <summary>
    /// The lag-augmented Gram matrix, built by forming every complete row and summing
    /// the outer products the long way round.
    /// </summary>
    public static double[,] Gram(double[] series, int lagDepth, double offset = 0d)
    {
        var dimension = lagDepth + 1;
        var gram = new double[dimension, dimension];

        for (var t = lagDepth; t < series.Length; t++)
        {
            for (var i = 0; i < dimension; i++)
            {
                for (var j = 0; j < dimension; j++)
                {
                    gram[i, j] += (series[t - i] - offset) * (series[t - j] - offset);
                }
            }
        }

        return gram;
    }

    /// <summary>Column sums of the same complete rows.</summary>
    public static double[] GramSums(double[] series, int lagDepth, double offset = 0d)
    {
        var sums = new double[lagDepth + 1];

        for (var t = lagDepth; t < series.Length; t++)
        {
            for (var i = 0; i <= lagDepth; i++)
            {
                sums[i] += series[t - i] - offset;
            }
        }

        return sums;
    }

    /// <summary>
    /// The classical divide-by-n autocovariances, straight from the textbook definition:
    /// a two-pass mean, then one centred sum per lag over its own natural range.
    /// </summary>
    public static double[] Autocovariance(double[] series, int maxLag)
    {
        var mean = 0d;
        foreach (var value in series)
        {
            mean += value;
        }

        mean /= series.Length;

        var gamma = new double[maxLag + 1];

        for (var k = 0; k <= maxLag; k++)
        {
            var total = 0d;

            for (var t = k; t < series.Length; t++)
            {
                total += (series[t] - mean) * (series[t - k] - mean);
            }

            gamma[k] = total / series.Length;
        }

        return gamma;
    }
}
