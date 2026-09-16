namespace TimeSeries.Transforms;

/// <summary>
/// One streaming <c>(1 - B^k)</c> difference. Holds a ring buffer of <c>k</c> values and
/// nothing else, so its memory is fixed by the lag rather than by the series length.
/// </summary>
/// <remarks>
/// This is the single place the library subtracts a lagged value. Both the composed
/// <see cref="DifferenceTransform"/> and the array entry points on
/// <see cref="Differencing"/> route through it, so there is one implementation of the
/// arithmetic and not two that can drift.
/// </remarks>
internal sealed class LagDifferenceStage
{
    private readonly double[] _window;
    private int _filled;
    private int _head;

    internal LagDifferenceStage(int lag)
    {
        Lag = lag;
        _window = new double[lag];
    }

    /// <summary>The lag <c>k</c> subtracted by this stage.</summary>
    internal int Lag { get; }

    /// <summary>True once <see cref="Lag"/> values have been consumed and output has begun.</summary>
    internal bool IsWarm => _filled == Lag;

    /// <summary>
    /// Differences one span. The first <see cref="Lag"/> values ever seen prime the ring
    /// buffer and produce no output; every value after that produces exactly one.
    /// </summary>
    /// <param name="source">Input values in time order.</param>
    /// <param name="destination">
    /// Destination, at least as long as <paramref name="source"/>. Writes trail reads, so
    /// the two may refer to the same buffer.
    /// </param>
    /// <returns>The number of values written.</returns>
    internal int Apply(ReadOnlySpan<double> source, Span<double> destination)
    {
        var written = 0;

        for (var i = 0; i < source.Length; i++)
        {
            var value = source[i];

            if (_filled < Lag)
            {
                _window[_filled++] = value;
                continue;
            }

            destination[written++] = value - _window[_head];
            _window[_head] = value;
            _head = _head + 1 == Lag ? 0 : _head + 1;
        }

        return written;
    }

    /// <summary>
    /// Copies the ring buffer out in chronological order, oldest first. These are the
    /// last <see cref="Lag"/> values this stage consumed, which is exactly what is needed
    /// to invert it.
    /// </summary>
    /// <param name="destination">A span of length <see cref="Lag"/>.</param>
    internal void CopyWindowTo(Span<double> destination)
    {
        for (var i = 0; i < Lag; i++)
        {
            var index = _head + i;
            destination[i] = _window[index >= Lag ? index - Lag : index];
        }
    }

    /// <summary>
    /// Fills the window from a saved chronological tail, oldest first, so the stage
    /// continues exactly where a previous traversal left off.
    /// </summary>
    /// <param name="window">Exactly <see cref="Lag"/> values.</param>
    internal void Prime(ReadOnlySpan<double> window)
    {
        window.CopyTo(_window);
        _filled = Lag;
        _head = 0;
    }

    /// <summary>Writes the complete stage state, warm or not.</summary>
    internal void WriteState(BinaryWriter writer)
    {
        writer.Write(Lag);
        writer.Write(_filled);
        writer.Write(_head);

        foreach (var value in _window)
        {
            writer.Write(value);
        }
    }

    /// <summary>Restores state written by <see cref="WriteState"/>.</summary>
    internal void ReadState(BinaryReader reader)
    {
        var lag = reader.ReadInt32();

        if (lag != Lag)
        {
            throw new InvalidDataException($"Saved differencing stage has lag {lag}; this one has {Lag}.");
        }

        _filled = reader.ReadInt32();
        _head = reader.ReadInt32();

        for (var i = 0; i < _window.Length; i++)
        {
            _window[i] = reader.ReadDouble();
        }
    }

    internal void Reset()
    {
        _filled = 0;
        _head = 0;
        Array.Clear(_window, 0, _window.Length);
    }
}
