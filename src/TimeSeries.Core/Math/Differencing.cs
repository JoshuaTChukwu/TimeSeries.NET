namespace TimeSeries.Core.Math;

public static class Differencing
{
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
