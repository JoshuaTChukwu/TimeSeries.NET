namespace TimeSeries.Accumulators;

/// <summary>
/// Accumulates the lag-augmented Gram matrix that both Hannan-Rissanen stages are solved
/// from, in one pass and in <c>(L+1)^2</c> memory regardless of series length.
/// </summary>
/// <remarks>
/// <para>
/// Only the upper triangle is updated in the inner loop — halving the multiply-adds per
/// row — and <see cref="Freeze"/> mirrors it into a full symmetric matrix. At
/// <c>L = 37</c> the state is about 11.6 KB whether the series has a thousand rows or
/// three hundred million.
/// </para>
/// <para>
/// Rows must be complete: the Gram is defined over <c>t in [L+1, n]</c>, and fixing both
/// estimation stages to that same range is what makes them mutually consistent. A
/// partitioned reader therefore primes its window with a warm-up prefix of <c>L</c> rows
/// and adds none of them.
/// </para>
/// </remarks>
public sealed class LagGramAccumulator : IAccumulator<LagGramAccumulator, GramMatrix>
{
    private readonly double[] _gram;
    private readonly double[]? _gramCompensation;
    private readonly double[] _sums;
    private readonly double[]? _sumCompensation;
    private readonly double[] _shifted;
    private double _count;
    private long _rows;

    /// <summary>
    /// Creates an accumulator over lags 0 through <paramref name="lagDepth"/>.
    /// </summary>
    /// <param name="lagDepth">
    /// The deepest lag <c>L</c>, normally <c>max(p, q + m)</c>. Zero or greater.
    /// </param>
    /// <param name="offset">
    /// A constant subtracted from every value before accumulation. Captured from the
    /// first observation of the series by the caller, not discovered here: partitions and
    /// folds have to agree on it for their states to be mergeable. Zero leaves values
    /// unshifted.
    /// </param>
    /// <param name="compensated">
    /// Enables Neumaier compensated summation, at roughly twice the flops. Worth it over
    /// the 10^8 additions a long series produces; unnecessary below that.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lagDepth"/> is negative.</exception>
    /// <exception cref="ArgumentException"><paramref name="offset"/> is not finite.</exception>
    public LagGramAccumulator(int lagDepth, double offset = 0d, bool compensated = false)
    {
        if (lagDepth < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(lagDepth), lagDepth, "Lag depth must be zero or greater.");
        }

        if (double.IsNaN(offset) || double.IsInfinity(offset))
        {
            throw new ArgumentException($"Offset must be finite; {offset} supplied.", nameof(offset));
        }

        LagDepth = lagDepth;
        Offset = offset;
        IsCompensated = compensated;

        var dimension = lagDepth + 1;
        _gram = new double[dimension * dimension];
        _sums = new double[dimension];
        _shifted = new double[dimension];

        if (compensated)
        {
            _gramCompensation = new double[dimension * dimension];
            _sumCompensation = new double[dimension];
        }
    }

    /// <summary>The deepest lag accumulated, <c>L</c>.</summary>
    public int LagDepth { get; }

    /// <summary>The required length of a row passed to <see cref="Add"/>, <c>L + 1</c>.</summary>
    public int Dimension => LagDepth + 1;

    /// <summary>The constant subtracted from every value before accumulation.</summary>
    public double Offset { get; }

    /// <summary>True when Neumaier compensated summation is in use.</summary>
    public bool IsCompensated { get; }

    /// <inheritdoc />
    public double Count => _count;

    /// <summary>The number of rows added, unaffected by <see cref="Scale"/>.</summary>
    public long Rows => _rows;

    /// <summary>
    /// Folds one complete lag row into the matrix.
    /// </summary>
    /// <param name="row">
    /// Exactly <see cref="Dimension"/> values: <c>row[0]</c> is <c>y_t</c> and
    /// <c>row[k]</c> is <c>y_(t-k)</c>.
    /// </param>
    /// <exception cref="ArgumentException">
    /// <paramref name="row"/> is not exactly <see cref="Dimension"/> long.
    /// </exception>
    public void Add(ReadOnlySpan<double> row)
    {
        if (row.Length != Dimension)
        {
            throw new ArgumentException(
                $"A lag row must hold exactly {Dimension} values for lag depth {LagDepth}; " +
                $"{row.Length} supplied. The Gram matrix is defined over complete rows only.",
                nameof(row));
        }

        var dimension = Dimension;
        var shifted = _shifted;

        for (var i = 0; i < dimension; i++)
        {
            shifted[i] = row[i] - Offset;
        }

        if (_gramCompensation is null)
        {
            for (var i = 0; i < dimension; i++)
            {
                var value = shifted[i];
                var start = i * dimension;

                for (var j = i; j < dimension; j++)
                {
                    _gram[start + j] += value * shifted[j];
                }

                _sums[i] += value;
            }
        }
        else
        {
            var sumCompensation = _sumCompensation!;

            for (var i = 0; i < dimension; i++)
            {
                var value = shifted[i];
                var start = i * dimension;

                for (var j = i; j < dimension; j++)
                {
                    Compensated.Add(
                        ref _gram[start + j], ref _gramCompensation[start + j], value * shifted[j]);
                }

                Compensated.Add(ref _sums[i], ref sumCompensation[i], value);
            }
        }

        _count++;
        _rows++;
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">
    /// The accumulators were built with a different lag depth or a different offset, so
    /// their states describe different quantities and cannot be added.
    /// </exception>
    public void Merge(in LagGramAccumulator other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        if (other.LagDepth != LagDepth)
        {
            throw new ArgumentException(
                $"Cannot merge a lag depth of {other.LagDepth} into one of {LagDepth}.",
                nameof(other));
        }

        if (!other.Offset.Equals(Offset))
        {
            throw new ArgumentException(
                $"Cannot merge accumulators with different offsets ({other.Offset} into {Offset}). " +
                "Partitions of one fit must be given the same offset for their states to add.",
                nameof(other));
        }

        MergeInto(_gram, _gramCompensation, other._gram, other._gramCompensation);
        MergeInto(_sums, _sumCompensation, other._sums, other._sumCompensation);

        _count += other._count;
        _rows += other._rows;
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

        ScaleInPlace(_gram, _gramCompensation, lambda);
        ScaleInPlace(_sums, _sumCompensation, lambda);
        _count *= lambda;
    }

    /// <inheritdoc />
    public GramMatrix Freeze()
    {
        var dimension = Dimension;
        var values = new double[dimension * dimension];

        for (var i = 0; i < dimension; i++)
        {
            var start = i * dimension;

            for (var j = i; j < dimension; j++)
            {
                var value = _gram[start + j] + (_gramCompensation?[start + j] ?? 0d);
                values[start + j] = value;
                values[(j * dimension) + i] = value;
            }
        }

        var sums = new double[dimension];

        for (var i = 0; i < dimension; i++)
        {
            sums[i] = _sums[i] + (_sumCompensation?[i] ?? 0d);
        }

        return new GramMatrix(LagDepth, values, sums, _count, _rows, Offset);
    }

    internal void WriteState(BinaryWriter writer)
    {
        writer.Write(Dimension);
        writer.Write(Offset);
        writer.Write(IsCompensated);
        writer.Write(_count);
        writer.Write(_rows);
        WriteArray(writer, _gram);
        WriteArray(writer, _sums);

        if (IsCompensated)
        {
            WriteArray(writer, _gramCompensation!);
            WriteArray(writer, _sumCompensation!);
        }
    }

    internal void ReadState(BinaryReader reader)
    {
        var dimension = reader.ReadInt32();
        var offset = reader.ReadDouble();
        var compensated = reader.ReadBoolean();

        if (dimension != Dimension || !offset.Equals(Offset) || compensated != IsCompensated)
        {
            throw new InvalidDataException(
                $"Saved Gram state (dimension {dimension}, offset {offset}, compensated {compensated}) does not " +
                $"match this accumulator (dimension {Dimension}, offset {Offset}, compensated {IsCompensated}).");
        }

        _count = reader.ReadDouble();
        _rows = reader.ReadInt64();
        ReadArray(reader, _gram);
        ReadArray(reader, _sums);

        if (IsCompensated)
        {
            ReadArray(reader, _gramCompensation!);
            ReadArray(reader, _sumCompensation!);
        }
    }

    private static void WriteArray(BinaryWriter writer, double[] values)
    {
        writer.Write(values.Length);

        foreach (var value in values)
        {
            writer.Write(value);
        }
    }

    private static void ReadArray(BinaryReader reader, double[] target)
    {
        var length = reader.ReadInt32();

        if (length != target.Length)
        {
            throw new InvalidDataException($"Saved array has {length} values; {target.Length} expected.");
        }

        for (var i = 0; i < length; i++)
        {
            target[i] = reader.ReadDouble();
        }
    }

    /// <summary>Discards all accumulated state, leaving the accumulator reusable.</summary>
    public void Reset()
    {
        Array.Clear(_gram, 0, _gram.Length);
        Array.Clear(_sums, 0, _sums.Length);
        _gramCompensation?.AsSpan().Clear();
        _sumCompensation?.AsSpan().Clear();
        _count = 0d;
        _rows = 0L;
    }

    private static void MergeInto(
        double[] target, double[]? targetCompensation, double[] source, double[]? sourceCompensation)
    {
        if (targetCompensation is null)
        {
            for (var i = 0; i < target.Length; i++)
            {
                target[i] += source[i] + (sourceCompensation?[i] ?? 0d);
            }

            return;
        }

        for (var i = 0; i < target.Length; i++)
        {
            Compensated.Add(ref target[i], ref targetCompensation[i], source[i]);

            if (sourceCompensation is not null)
            {
                Compensated.Add(ref target[i], ref targetCompensation[i], sourceCompensation[i]);
            }
        }
    }

    private static void ScaleInPlace(double[] values, double[]? compensation, double lambda)
    {
        for (var i = 0; i < values.Length; i++)
        {
            values[i] *= lambda;
        }

        if (compensation is null)
        {
            return;
        }

        for (var i = 0; i < compensation.Length; i++)
        {
            compensation[i] *= lambda;
        }
    }
}
