namespace TimeSeries.Accumulators;

/// <summary>
/// Accumulates count, mean, variance and range in one pass, in constant memory.
/// </summary>
/// <remarks>
/// <para>
/// Implemented as offset-shifted power sums rather than as a Welford recurrence. Welford
/// is the usual choice because it avoids the cancellation in <c>sum x^2 / n - mean^2</c>,
/// but the offset already does that: values are shifted by a constant near the data
/// before anything is squared, so the sums stay small and well conditioned.
/// </para>
/// <para>
/// What the power sums add is that <see cref="Merge"/> is plain addition, and therefore
/// exact. Welford's parallel merge is not — it divides by a combined count — so
/// partitioned and single-pass fits would disagree in the last bits, and the
/// exact-merge invariant this library asserts everywhere else would have one exception
/// in it. That trade is worth naming rather than burying.
/// </para>
/// </remarks>
public sealed class MomentsAccumulator : IAccumulator<MomentsAccumulator, Moments>
{
    private double _sum;
    private double _sumOfSquares;
    private double _count;
    private double _minimum = double.PositiveInfinity;
    private double _maximum = double.NegativeInfinity;

    /// <summary>
    /// Creates an accumulator.
    /// </summary>
    /// <param name="offset">
    /// A constant subtracted from every value before accumulation. Partitions of one fit
    /// must share it for their states to be mergeable.
    /// </param>
    /// <exception cref="ArgumentException"><paramref name="offset"/> is not finite.</exception>
    public MomentsAccumulator(double offset = 0d)
    {
        if (double.IsNaN(offset) || double.IsInfinity(offset))
        {
            throw new ArgumentException($"Offset must be finite; {offset} supplied.", nameof(offset));
        }

        Offset = offset;
    }

    /// <summary>The constant subtracted from every value before accumulation.</summary>
    public double Offset { get; }

    /// <inheritdoc />
    public double Count => _count;

    /// <summary>
    /// Folds the current observation in.
    /// </summary>
    /// <param name="row">
    /// A lag row, of which only <c>row[0]</c> — the current observation — is read. Taking
    /// a row rather than a scalar is what lets one fit loop feed this accumulator, the
    /// Gram and the autocovariances from a single shared lag window.
    /// </param>
    /// <exception cref="ArgumentException"><paramref name="row"/> is empty.</exception>
    public void Add(ReadOnlySpan<double> row)
    {
        if (row.Length == 0)
        {
            throw new ArgumentException("A lag row must hold at least the current observation.", nameof(row));
        }

        Add(row[0]);
    }

    /// <summary>Folds one observation in.</summary>
    /// <param name="value">The observation, on the original scale.</param>
    public void Add(double value)
    {
        var shifted = value - Offset;

        _sum += shifted;
        _sumOfSquares += shifted * shifted;
        _count++;

        if (shifted < _minimum)
        {
            _minimum = shifted;
        }

        if (shifted > _maximum)
        {
            _maximum = shifted;
        }
    }

    /// <summary>Folds a whole batch of observations in.</summary>
    /// <param name="values">Observations on the original scale.</param>
    public void AddRange(ReadOnlySpan<double> values)
    {
        for (var i = 0; i < values.Length; i++)
        {
            Add(values[i]);
        }
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The accumulators were built with different offsets.</exception>
    public void Merge(in MomentsAccumulator other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        if (!other.Offset.Equals(Offset))
        {
            throw new ArgumentException(
                $"Cannot merge accumulators with different offsets ({other.Offset} into {Offset}).",
                nameof(other));
        }

        _sum += other._sum;
        _sumOfSquares += other._sumOfSquares;
        _count += other._count;

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

        _sum *= lambda;
        _sumOfSquares *= lambda;
        _count *= lambda;
    }

    /// <inheritdoc />
    public Moments Freeze()
    {
        if (_count == 0d)
        {
            return new Moments(0d, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN);
        }

        // An exactly constant series is settled here rather than by arithmetic. The
        // subtraction below is a difference of two nearly equal large numbers, and at a
        // level such as 1234.5678 it returns rounding noise — a variance around 1e-8
        // where the answer is zero. Downstream that noise is not harmless: the standing
        // decision to detect a constant differenced series and short-circuit it depends
        // on the variance actually being zero. The range is already tracked, so the
        // exact test is free.
        if (_maximum.Equals(_minimum))
        {
            var constant = _minimum + Offset;

            return new Moments(
                _count, constant, 0d, _count > 1d ? 0d : double.NaN, constant, constant);
        }

        var mean = _sum / _count;
        var centred = _sumOfSquares - (_sum * mean);

        // Cancellation can also push a near-constant series a hair below zero.
        if (centred < 0d)
        {
            centred = 0d;
        }

        var variance = centred / _count;
        var sampleVariance = _count > 1d ? centred / (_count - 1d) : double.NaN;

        return new Moments(
            _count,
            mean + Offset,
            variance,
            sampleVariance,
            _minimum + Offset,
            _maximum + Offset);
    }

    internal void WriteState(BinaryWriter writer)
    {
        writer.Write(Offset);
        writer.Write(_sum);
        writer.Write(_sumOfSquares);
        writer.Write(_count);
        writer.Write(_minimum);
        writer.Write(_maximum);
    }

    internal void ReadState(BinaryReader reader)
    {
        var offset = reader.ReadDouble();

        if (!offset.Equals(Offset))
        {
            throw new InvalidDataException($"Saved moments have offset {offset}; this accumulator has {Offset}.");
        }

        _sum = reader.ReadDouble();
        _sumOfSquares = reader.ReadDouble();
        _count = reader.ReadDouble();
        _minimum = reader.ReadDouble();
        _maximum = reader.ReadDouble();
    }

    /// <summary>Discards all accumulated state, leaving the accumulator reusable.</summary>
    public void Reset()
    {
        _sum = 0d;
        _sumOfSquares = 0d;
        _count = 0d;
        _minimum = double.PositiveInfinity;
        _maximum = double.NegativeInfinity;
    }
}
