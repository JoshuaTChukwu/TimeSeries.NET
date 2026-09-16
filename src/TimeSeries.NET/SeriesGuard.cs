namespace TimeSeries;

/// <summary>Shared argument checks for series-shaped inputs.</summary>
internal static class SeriesGuard
{
    /// <summary>Throws <see cref="InvalidSeriesException"/> at the first non-finite value.</summary>
    internal static void Finite(ReadOnlySpan<double> values, string paramName, long indexOffset = 0)
    {
        for (var i = 0; i < values.Length; i++)
        {
            var value = values[i];

            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                throw new InvalidSeriesException(paramName, indexOffset + i, value);
            }
        }
    }

    /// <summary>Throws when two paired spans differ in length or are empty.</summary>
    internal static void SameNonEmptyLength(
        ReadOnlySpan<double> first, string firstName, ReadOnlySpan<double> second, string secondName)
    {
        if (first.Length == 0)
        {
            throw new ArgumentException("At least one observation is required.", firstName);
        }

        if (first.Length != second.Length)
        {
            throw new ArgumentException(
                $"'{firstName}' has {first.Length} values but '{secondName}' has {second.Length}; " +
                "they must be the same length.",
                secondName);
        }
    }
}
