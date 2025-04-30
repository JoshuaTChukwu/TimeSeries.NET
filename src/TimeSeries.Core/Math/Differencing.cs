namespace TimeSeries.Core.Math;

/// <summary>
/// Provides methods for differencing a time series to remove trend and make it stationary.
/// </summary>
public static class Differencing
{
    /// <summary>
    /// Computes the lag-based difference of a time series to help remove trends and prepare the data for stationarity analysis.
    /// </summary>
    /// <param name="data">The input time series array.</param>
    /// <param name="lag">The number of time steps to subtract (default is 1).</param>
    /// <param name="isCumulative">
    /// If true, computes the cumulative sum of the differences (useful for reconstructing cumulative trends).
    /// </param>
    /// <returns>
    /// A new array containing the differenced values. The length will be (data.Length - lag).
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown if the lag is less than 1 or if the input series is null or too short for the specified lag.
    /// </exception>

    public static double[] Difference(double[] data, int lag = 1, bool isCumulative = false)
    {
        if (lag < 1)
            throw new ArgumentException("Lag must be greater than 0.");
        if (data == null || data.Length <= lag)
            throw new ArgumentException("Series must be longer than lag.");

        double[] result = new double[data.Length - lag];
        for (int i = lag; i < data.Length; i++)
        {
            result[i - lag] = data[i] - data[i - lag];
        }

        if (isCumulative)
        {
            for (int i = 1; i < result.Length; i++)
            {
                result[i] += result[i - 1];
            }
        }
        return result;
    }

    /// <summary>
    /// Reconstructs the original time series from differenced data using a provided base segment of the original series.
    /// Supports cumulative and log-transformed restoration.
    /// </summary>
    /// <param name="differencedData">The differenced series (e.g., output from first-order differencing).</param>
    /// <param name="originalData">The original time series used to seed the initial lag values for reconstruction.</param>
    /// <param name="lag">The lag used during differencing (typically 1).</param>
    /// <param name="startIndex">
    /// The index in the original data to begin seeding from. Default is 0.
    /// </param>
    /// <param name="isCumulative">
    /// If true, treats the differenced data as a cumulative sum before applying inverse differencing.
    /// </param>
    /// <param name="isLog">
    /// If true, applies exponential transformation to the restored series (for log-transformed data).
    /// </param>
    /// <param name="validateAgainstOriginal">
    /// If true, checks whether the reconstructed values match the original series. Only valid when <paramref name="isCumulative"/> and <paramref name="isLog"/> are false.
    /// </param>
    /// <returns>A reconstructed series of the same length as (differencedData.Length + lag).</returns>
    /// <exception cref="ArgumentException">
    /// Thrown if lag is less than 1, or if the original or differenced arrays are null or improperly sized.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when validation fails (reconstructed values do not match the original).
    /// </exception>


    public static double[] InverseDifference(
     double[] differencedData,
     double[] originalData,
     int lag,
     int startIndex = 0,
     bool isCumulative = false,
     bool isLog = false,
     bool validateAgainstOriginal = false)
    {
        if (lag < 1)
            throw new ArgumentException("Lag must be greater than 0.");
        if (differencedData == null || originalData == null)
            throw new ArgumentNullException("Inputs cannot be null.");
        if (originalData.Length - startIndex < lag)
            throw new ArgumentException("Not enough original data to restore.");
        if (differencedData.Length + lag + startIndex > originalData.Length && validateAgainstOriginal)
            throw new ArgumentException("Differenced and original data sizes are mismatched.");

        int n = differencedData.Length;

        // Apply cumulative if needed before restoration
        double[] processedDiff = (double[])differencedData.Clone();
        if (isCumulative)
        {
            for (int i = 1; i < processedDiff.Length; i++)
            {
                processedDiff[i] += processedDiff[i - 1];
            }
        }

        double[] restored = new double[n + lag];

        // Step 1: Seed with original lag values
        for (int i = 0; i < lag; i++)
        {
            restored[i] = originalData[startIndex + i];
        }

        // Step 2: Restore with processed diff
        for (int i = lag; i < restored.Length; i++)
        {
            restored[i] = restored[i - lag] + processedDiff[i - lag];
        }

        // Step 3: Apply exponential if needed
        if (isLog)
        {
            for (int i = 0; i < restored.Length; i++)
            {
                restored[i] = System.Math.Exp(restored[i]);
            }
        }

        // Step 4: Validate only if not transformed
        if (validateAgainstOriginal && !isCumulative && !isLog)
        {
            for (int i = 0; i < restored.Length; i++)
            {
                if (System.Math.Abs(restored[i] - originalData[startIndex + i]) > 1e-6)
                    throw new InvalidOperationException("Restored data does not match the original.");
            }
        }

        return restored;
    }


}
