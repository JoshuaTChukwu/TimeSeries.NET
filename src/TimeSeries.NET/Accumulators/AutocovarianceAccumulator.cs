namespace TimeSeries.Accumulators;

/// <summary>
/// Accumulates the classical autocovariance sequence alongside the Gram matrix, sharing
/// the same lag window and costing a handful of running sums.
/// </summary>
/// <remarks>
/// <para>
/// This exists as a separate accumulator because <see cref="GramMatrix"/> is not
/// Toeplitz: its entries at equal lag distance cover row ranges offset by one, so they
/// differ. Durbin-Levinson needs a genuinely Toeplitz sequence, and the divide-by-n
/// estimator is positive semi-definite precisely because each <c>gamma-hat_k</c> is
/// summed over its own natural range <c>t in [k+1, n]</c> and divided by the same
/// <c>n</c>. Deriving the autocovariances from <c>G</c> instead would quietly destroy
/// that property.
/// </para>
/// <para>
/// Unlike the Gram accumulator, partial rows are welcome: an observation early in the
/// series contributes to every lag it has a partner for and to no others, which is
/// exactly what the classical ranges require.
/// </para>
/// </remarks>
public sealed class AutocovarianceAccumulator
    : IAccumulator<AutocovarianceAccumulator, Autocovariances>
{
    private readonly double[] _crossProducts;
    private readonly double[] _leadSums;
    private readonly double[] _lagSums;
    private readonly double[] _counts;
    private double _minimum = double.PositiveInfinity;
    private double _maximum = double.NegativeInfinity;

    /// <summary>
    /// Creates an accumulator over lags 0 through <paramref name="maxLag"/>.
    /// </summary>
    /// <param name="maxLag">The deepest lag to estimate. Zero or greater.</param>
    /// <param name="offset">
    /// A constant subtracted from every value before accumulation, to keep a large level
    /// from swamping the deviations. The autocovariances are invariant to it in exact
    /// arithmetic and far better conditioned with it. Partitions of one fit must share it.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is negative.</exception>
    /// <exception cref="ArgumentException"><paramref name="offset"/> is not finite.</exception>
    public AutocovarianceAccumulator(int maxLag, double offset = 0d)
    {
        if (maxLag < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maxLag), maxLag, "Maximum lag must be zero or greater.");
        }

        if (double.IsNaN(offset) || double.IsInfinity(offset))
        {
            throw new ArgumentException($"Offset must be finite; {offset} supplied.", nameof(offset));
        }

        MaxLag = maxLag;
        Offset = offset;

        var length = maxLag + 1;
        _crossProducts = new double[length];
        _leadSums = new double[length];
        _lagSums = new double[length];
        _counts = new double[length];
    }

    /// <summary>The deepest lag estimated, <c>L</c>.</summary>
    public int MaxLag { get; }

    /// <summary>The constant subtracted from every value before accumulation.</summary>
    public double Offset { get; }

    /// <inheritdoc />
    /// <remarks>
    /// The lag-0 count, which is the number of observations seen — every observation
    /// contributes to lag 0 and to shallower lags only as partners become available.
    /// </remarks>
    public double Count => _counts[0];

    /// <summary>
    /// Folds one observation and whatever lagged partners it has into the sums.
    /// </summary>
    /// <param name="row">
    /// The lag row ending at the current observation: <c>row[0]</c> is <c>y_t</c> and
    /// <c>row[k]</c> is <c>y_(t-k)</c>. May be shorter than <c>MaxLag + 1</c> early in a
    /// series, in which case the missing lags are simply not contributed to.
    /// </param>
    /// <exception cref="ArgumentException">
    /// <paramref name="row"/> is empty, or longer than <c>MaxLag + 1</c>.
    /// </exception>
    public void Add(ReadOnlySpan<double> row)
    {
        if (row.Length == 0)
        {
            throw new ArgumentException("A lag row must hold at least the current observation.", nameof(row));
        }

        if (row.Length > MaxLag + 1)
        {
            throw new ArgumentException(
                $"A lag row may hold at most {MaxLag + 1} values for a maximum lag of {MaxLag}; " +
                $"{row.Length} supplied.",
                nameof(row));
        }

        var lead = row[0] - Offset;

        if (lead < _minimum)
        {
            _minimum = lead;
        }

        if (lead > _maximum)
        {
            _maximum = lead;
        }

        for (var k = 0; k < row.Length; k++)
        {
            var lagged = row[k] - Offset;

            _crossProducts[k] += lead * lagged;
            _leadSums[k] += lead;
            _lagSums[k] += lagged;
            _counts[k]++;
        }
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">
    /// The accumulators were built with a different maximum lag or a different offset.
    /// </exception>
    public void Merge(in AutocovarianceAccumulator other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        if (other.MaxLag != MaxLag)
        {
            throw new ArgumentException(
                $"Cannot merge a maximum lag of {other.MaxLag} into one of {MaxLag}.", nameof(other));
        }

        if (!other.Offset.Equals(Offset))
        {
            throw new ArgumentException(
                $"Cannot merge accumulators with different offsets ({other.Offset} into {Offset}).",
                nameof(other));
        }

        for (var k = 0; k <= MaxLag; k++)
        {
            _crossProducts[k] += other._crossProducts[k];
            _leadSums[k] += other._leadSums[k];
            _lagSums[k] += other._lagSums[k];
            _counts[k] += other._counts[k];
        }

        if (other._minimum < _minimum)
        {
            _minimum = other._minimum;
        }

        if (other._maximum > _maximum)
        {
            _maximum = other._maximum;
        }
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lambda"/> is outside (0, 1].</exception>
    public void Scale(double lambda)
    {
        if (!(lambda > 0d && lambda <= 1d))
        {
            throw new ArgumentOutOfRangeException(
                nameof(lambda), lambda, "The forgetting factor must lie in (0, 1].");
        }

        if (lambda == 1d)
        {
            return;
        }

        for (var k = 0; k <= MaxLag; k++)
        {
            _crossProducts[k] *= lambda;
            _leadSums[k] *= lambda;
            _lagSums[k] *= lambda;
            _counts[k] *= lambda;
        }
    }

    /// <inheritdoc />
    public Autocovariances Freeze()
    {
        var length = MaxLag + 1;
        var gamma = new double[length];
        var acf = new double[length];
        var n = _counts[0];

        if (n == 0d)
        {
            return new Autocovariances(MaxLag, gamma, acf, double.NaN, 0d);
        }

        // An exactly constant series is settled from the range rather than by arithmetic.
        // The expansion below is a difference of nearly equal large numbers, and at a
        // level such as 1234.5678 it returns rounding noise: a variance around 1e-8 where
        // the answer is zero, which then divides into itself to produce an autocorrelation
        // near 0.98 out of a series that never moves. That is a fabricated result, not an
        // imprecise one, so it is cut off before it can be formed.
        if (_maximum.Equals(_minimum))
        {
            acf[0] = 1d;
            return new Autocovariances(MaxLag, gamma, acf, _minimum + Offset, n);
        }

        // The centred sum expanded so it can be accumulated in one pass:
        //   sum (y_t - m)(y_(t-k) - m)
        //     = sum y_t y_(t-k)  -  m * sum y_t  -  m * sum y_(t-k)  +  count_k * m^2
        var mean = _leadSums[0] / n;

        for (var k = 0; k < length; k++)
        {
            var centred = _crossProducts[k]
                - (mean * _leadSums[k])
                - (mean * _lagSums[k])
                + (_counts[k] * mean * mean);

            gamma[k] = centred / n;
        }

        var variance = gamma[0];

        if (variance > 0d)
        {
            for (var k = 0; k < length; k++)
            {
                acf[k] = gamma[k] / variance;
            }
        }
        else
        {
            // A constant series has no autocorrelation to speak of. Reporting 1 at lag 0
            // and 0 elsewhere keeps callers off a 0/0, and the estimator short-circuits
            // this case before it ever gets here.
            acf[0] = 1d;
        }

        return new Autocovariances(MaxLag, gamma, acf, mean + Offset, n);
    }

    /// <summary>Discards all accumulated state, leaving the accumulator reusable.</summary>
    public void Reset()
    {
        Array.Clear(_crossProducts, 0, _crossProducts.Length);
        Array.Clear(_leadSums, 0, _leadSums.Length);
        Array.Clear(_lagSums, 0, _lagSums.Length);
        Array.Clear(_counts, 0, _counts.Length);
        _minimum = double.PositiveInfinity;
        _maximum = double.NegativeInfinity;
    }
}
