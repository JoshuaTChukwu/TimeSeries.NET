namespace TimeSeries.Transforms;

/// <summary>
/// Array entry points for differencing and integration.
/// </summary>
/// <remarks>
/// <para>
/// These are conveniences over <see cref="DifferenceTransform"/> and
/// <see cref="IntegrationState"/>, not a second implementation: every subtraction in the
/// library happens in one place, so the array path and the streaming path cannot drift.
/// </para>
/// <para>
/// Note the ceiling. A <c>double[]</c> tops out at 268,435,456 elements against the .NET
/// 2 GB object limit, so anything larger has to go through the streaming path.
/// </para>
/// </remarks>
public static class Differencing
{
    /// <summary>
    /// Applies <c>(1-B)^d (1-B^s)^D</c> to a series.
    /// </summary>
    /// <param name="series">The input series, in time order.</param>
    /// <param name="spec">The differencing to apply.</param>
    /// <returns>
    /// A new array of length <c>series.Length - spec.WarmupLength</c>, empty when the
    /// series is no longer than the warm-up.
    /// </returns>
    public static double[] Difference(ReadOnlySpan<double> series, DifferenceSpec spec)
    {
        var length = Math.Max(0, series.Length - spec.WarmupLength);
        var result = new double[length];

        if (length == 0)
        {
            return result;
        }

        // Fed in chunks against a fixed scratch buffer. A transform needs an output span
        // at least as long as its input, and the result is always shorter than the series
        // once warm-up is non-zero; chunking pays for that with 8 KB rather than with a
        // second copy of the series.
        const int ChunkSize = 1024;
        var transform = new DifferenceTransform(spec);
        var scratch = new double[ChunkSize];
        var written = 0;

        for (var start = 0; start < series.Length; start += ChunkSize)
        {
            var take = Math.Min(ChunkSize, series.Length - start);
            var produced = transform.Transform(series.Slice(start, take), scratch);
            scratch.AsSpan(0, produced).CopyTo(result.AsSpan(written));
            written += produced;
        }

        return result;
    }

    /// <summary>
    /// Applies <c>(1-B)^d (1-B^s)^D</c> to a series and captures the state that inverts it.
    /// </summary>
    /// <param name="series">The input series, in time order.</param>
    /// <param name="spec">The differencing to apply.</param>
    /// <param name="state">
    /// Receives the state seeded from the head of <paramref name="series"/>. Passing the
    /// result of this method and this state to
    /// <see cref="Integrate(ReadOnlySpan{double}, IntegrationState)"/> reproduces
    /// <paramref name="series"/> from its warm-up onwards.
    /// </param>
    /// <returns>The differenced series.</returns>
    /// <exception cref="ArgumentException">
    /// <paramref name="series"/> is shorter than the warm-up, so no state can be captured.
    /// </exception>
    public static double[] Difference(
        ReadOnlySpan<double> series, DifferenceSpec spec, out IntegrationState state)
    {
        state = IntegrationState.FromStart(series, spec);
        return Difference(series, spec);
    }

    /// <summary>
    /// Integrates a differenced series back to its original scale.
    /// </summary>
    /// <param name="differenced">Values on the differenced scale, in time order.</param>
    /// <param name="state">The state captured when the series was differenced.</param>
    /// <returns>A new array of the same length, on the original scale.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="state"/> is null.</exception>
    public static double[] Integrate(ReadOnlySpan<double> differenced, IntegrationState state)
    {
        if (state is null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        return state.Integrate(differenced);
    }

    /// <summary>
    /// Computes the lag-based difference of a time series to help remove trends and
    /// prepare the data for stationarity analysis.
    /// </summary>
    /// <param name="data">The input time series array.</param>
    /// <param name="lag">The number of time steps to subtract (default is 1).</param>
    /// <param name="isCumulative">
    /// If true, computes the cumulative sum of the differences (useful for reconstructing
    /// cumulative trends).
    /// </param>
    /// <returns>
    /// A new array containing the differenced values. The length will be
    /// (data.Length - lag).
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown if the lag is less than 1 or if the input series is null or too short for
    /// the specified lag.
    /// </exception>
    public static double[] Difference(double[] data, int lag = 1, bool isCumulative = false)
    {
        if (lag < 1)
        {
            throw new ArgumentException("Lag must be greater than 0.");
        }

        if (data == null || data.Length <= lag)
        {
            throw new ArgumentException("Series must be longer than lag.");
        }

        var result = new double[data.Length - lag];
        var stage = new LagDifferenceStage(lag);
        stage.Apply(data, result);

        if (isCumulative)
        {
            for (var i = 1; i < result.Length; i++)
            {
                result[i] += result[i - 1];
            }
        }

        return result;
    }

    /// <summary>
    /// Reconstructs the original time series from differenced data using a provided base
    /// segment of the original series. Supports cumulative and log-transformed restoration.
    /// </summary>
    /// <param name="differencedData">
    /// The differenced series (e.g., output from first-order differencing).
    /// </param>
    /// <param name="originalData">
    /// The original time series used to seed the initial lag values for reconstruction.
    /// </param>
    /// <param name="lag">The lag used during differencing (typically 1).</param>
    /// <param name="startIndex">
    /// The index in the original data to begin seeding from. Default is 0.
    /// </param>
    /// <param name="isCumulative">
    /// If true, treats the differenced data as a cumulative sum before applying inverse
    /// differencing.
    /// </param>
    /// <param name="isLog">
    /// If true, applies exponential transformation to the restored series (for
    /// log-transformed data).
    /// </param>
    /// <param name="validateAgainstOriginal">
    /// If true, checks whether the reconstructed values match the original series. Only
    /// valid when <paramref name="isCumulative"/> and <paramref name="isLog"/> are false.
    /// </param>
    /// <returns>
    /// A reconstructed series of the same length as (differencedData.Length + lag).
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown if either input array is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown if lag is less than 1, or if the original or differenced arrays are
    /// improperly sized.
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
        {
            throw new ArgumentException("Lag must be greater than 0.");
        }

        if (differencedData == null)
        {
            throw new ArgumentNullException(nameof(differencedData));
        }

        if (originalData == null)
        {
            throw new ArgumentNullException(nameof(originalData));
        }

        if (originalData.Length - startIndex < lag)
        {
            throw new ArgumentException("Not enough original data to restore.");
        }

        if (differencedData.Length + lag + startIndex > originalData.Length && validateAgainstOriginal)
        {
            throw new ArgumentException("Differenced and original data sizes are mismatched.");
        }

        var n = differencedData.Length;

        // Apply cumulative if needed before restoration
        var processedDiff = (double[])differencedData.Clone();
        if (isCumulative)
        {
            for (var i = 1; i < processedDiff.Length; i++)
            {
                processedDiff[i] += processedDiff[i - 1];
            }
        }

        var restored = new double[n + lag];

        // Step 1: Seed with original lag values
        for (var i = 0; i < lag; i++)
        {
            restored[i] = originalData[startIndex + i];
        }

        // Step 2: Restore with processed diff
        for (var i = lag; i < restored.Length; i++)
        {
            restored[i] = restored[i - lag] + processedDiff[i - lag];
        }

        // Step 3: Apply exponential if needed
        if (isLog)
        {
            for (var i = 0; i < restored.Length; i++)
            {
                restored[i] = Math.Exp(restored[i]);
            }
        }

        // Step 4: Validate only if not transformed
        if (validateAgainstOriginal && !isCumulative && !isLog)
        {
            for (var i = 0; i < restored.Length; i++)
            {
                if (Math.Abs(restored[i] - originalData[startIndex + i]) > 1e-6)
                {
                    throw new InvalidOperationException("Restored data does not match the original.");
                }
            }
        }

        return restored;
    }
}
