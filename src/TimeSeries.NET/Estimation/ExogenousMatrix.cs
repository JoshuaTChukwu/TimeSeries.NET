using TimeSeries.Data;

namespace TimeSeries;

/// <summary>
/// Exogenous regressors aligned with a series: <c>Count</c> rows of <c>RegressorCount</c>
/// values, stored row-major.
/// </summary>
public sealed class ExogenousMatrix
{
    private readonly ReadOnlyMemory<double> _values;

    /// <summary>
    /// Wraps row-major values.
    /// </summary>
    /// <param name="rowMajor">
    /// <c>Count * RegressorCount</c> values, row after row. The memory is referenced, not copied.
    /// </param>
    /// <param name="regressorCount">The number of regressors per row. At least 1.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="regressorCount"/> is below 1.</exception>
    /// <exception cref="ArgumentException">
    /// The length of <paramref name="rowMajor"/> is not a multiple of <paramref name="regressorCount"/>.
    /// </exception>
    public ExogenousMatrix(ReadOnlyMemory<double> rowMajor, int regressorCount)
    {
        if (regressorCount < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(regressorCount), regressorCount, "At least one regressor is required.");
        }

        if (rowMajor.Length % regressorCount != 0)
        {
            throw new ArgumentException(
                $"{rowMajor.Length} values do not form whole rows of {regressorCount} regressors.",
                nameof(rowMajor));
        }

        _values = rowMajor;
        RegressorCount = regressorCount;
        Count = rowMajor.Length / regressorCount;
    }

    /// <summary>The number of rows, equal to the length of the series they accompany.</summary>
    public int Count { get; }

    /// <summary>The number of regressors per row.</summary>
    public int RegressorCount { get; }

    /// <summary>All values, row-major.</summary>
    public ReadOnlyMemory<double> Values => _values;

    /// <summary>Reads one value.</summary>
    /// <param name="row">Zero-based row, aligned with the series index.</param>
    /// <param name="regressor">Zero-based regressor.</param>
    /// <returns>The value at that position.</returns>
    public double this[int row, int regressor] => _values.Span[(row * RegressorCount) + regressor];

    /// <summary>One row of regressors.</summary>
    /// <param name="row">Zero-based row.</param>
    /// <returns>A span of <see cref="RegressorCount"/> values.</returns>
    public ReadOnlySpan<double> Row(int row) => _values.Span.Slice(row * RegressorCount, RegressorCount);

    /// <summary>
    /// The rows <c>[start, start + count)</c>, still row-major.
    /// </summary>
    /// <param name="start">First row.</param>
    /// <param name="count">Number of rows.</param>
    /// <returns>A span of <c>count * RegressorCount</c> values.</returns>
    public ReadOnlySpan<double> Rows(int start, int count)
        => _values.Span.Slice(start * RegressorCount, count * RegressorCount);

    /// <summary>
    /// Builds a matrix from one array per regressor.
    /// </summary>
    /// <param name="columns">Equal-length arrays, one per regressor, each aligned with the series.</param>
    /// <returns>The matrix, with the values copied into row-major order.</returns>
    /// <exception cref="ArgumentException">No columns, or columns of unequal length.</exception>
    public static ExogenousMatrix FromColumns(params double[][] columns)
    {
        if (columns is null || columns.Length == 0)
        {
            throw new ArgumentException("At least one column is required.", nameof(columns));
        }

        var count = columns[0].Length;

        for (var i = 1; i < columns.Length; i++)
        {
            if (columns[i].Length != count)
            {
                throw new ArgumentException(
                    $"Column {i} has {columns[i].Length} values but column 0 has {count}; all columns must align.",
                    nameof(columns));
            }
        }

        var values = new double[count * columns.Length];

        for (var row = 0; row < count; row++)
        {
            for (var i = 0; i < columns.Length; i++)
            {
                values[(row * columns.Length) + i] = columns[i][row];
            }
        }

        return new ExogenousMatrix(values, columns.Length);
    }

    /// <summary>
    /// Reads every batch of a source and keeps only its regressors.
    /// </summary>
    /// <remarks>
    /// This is how the future regressor values an ARIMAX forecast needs come out of the
    /// consumer's database: a scenario table or view holding the assumed paths — a
    /// <c>DbSeriesQuery</c> with no <c>ValueColumn</c> — read through the same adapter as
    /// the history. The library never extrapolates regressors; the scenario is the
    /// consumer's statement of what they assume.
    /// </remarks>
    /// <param name="source">A source with at least one regressor. Its values, if any, are ignored.</param>
    /// <param name="cancellationToken">Checked at every batch boundary.</param>
    /// <returns>The regressors, row for row, in scan order.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="source"/> carries no regressors.</exception>
    public static async ValueTask<ExogenousMatrix> FromSourceAsync(ITimeSeriesSource source, CancellationToken cancellationToken = default)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (source.RegressorCount < 1)
        {
            throw new ArgumentException("The source carries no regressors.", nameof(source));
        }

        var values = new List<double>();
        var cursor = await source.OpenAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            while (await cursor.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                values.AddRange(cursor.Current.Exogenous.ToArray());
            }
        }
        finally
        {
            await cursor.DisposeAsync().ConfigureAwait(false);
        }

        return new ExogenousMatrix(values.ToArray(), source.RegressorCount);
    }

    /// <summary>
    /// Builds a matrix from a single regressor.
    /// </summary>
    /// <param name="column">The regressor, aligned with the series.</param>
    /// <returns>A one-regressor matrix over the same memory.</returns>
    public static ExogenousMatrix FromColumn(double[] column)
    {
        if (column is null)
        {
            throw new ArgumentNullException(nameof(column));
        }

        return new ExogenousMatrix(column, 1);
    }
}
