namespace TimeSeries.Accumulators;

/// <summary>
/// A ring buffer that turns a stream of observations into lag rows
/// <c>v_t = (y_t, y_(t-1), ..., y_(t-L))</c>.
/// </summary>
/// <remarks>
/// One window feeds every accumulator in a fit. That sharing is the reason the classical
/// autocovariances cost only a handful of running sums on top of the Gram matrix: the
/// lagged values they both need have already been buffered.
/// </remarks>
public sealed class LagWindow
{
    private readonly double[] _buffer;
    private int _next;
    private int _seen;

    /// <summary>
    /// Creates a window over lags 0 through <paramref name="lagDepth"/>.
    /// </summary>
    /// <param name="lagDepth">The deepest lag <c>L</c> the window must retain. Zero or greater.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lagDepth"/> is negative.</exception>
    public LagWindow(int lagDepth)
    {
        if (lagDepth < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(lagDepth), lagDepth, "Lag depth must be zero or greater.");
        }

        LagDepth = lagDepth;
        _buffer = new double[lagDepth + 1];
    }

    /// <summary>The deepest lag retained, <c>L</c>.</summary>
    public int LagDepth { get; }

    /// <summary>The length of a full lag row, <c>L + 1</c>.</summary>
    public int RowLength => LagDepth + 1;

    /// <summary>
    /// The number of lag values currently available, which is <see cref="RowLength"/>
    /// once <see cref="RowLength"/> observations have been pushed and fewer before that.
    /// </summary>
    public int Available => _seen < RowLength ? _seen : RowLength;

    /// <summary>True once a full lag row is available.</summary>
    public bool IsFull => _seen >= RowLength;

    /// <summary>Pushes the next observation in time order.</summary>
    /// <param name="value">The observation <c>y_t</c>.</param>
    public void Push(double value)
    {
        _buffer[_next] = value;
        _next = _next + 1 == RowLength ? 0 : _next + 1;

        if (_seen < RowLength)
        {
            _seen++;
        }
    }

    /// <summary>
    /// Copies the lag row ending at the most recently pushed observation.
    /// </summary>
    /// <param name="destination">
    /// Destination of at least <see cref="Available"/> values. <c>destination[0]</c>
    /// receives <c>y_t</c> and <c>destination[k]</c> receives <c>y_(t-k)</c>.
    /// </param>
    /// <returns>The number of values written, equal to <see cref="Available"/>.</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is too short.</exception>
    /// <exception cref="InvalidOperationException">Nothing has been pushed yet.</exception>
    public int CopyRow(Span<double> destination)
    {
        if (_seen == 0)
        {
            throw new InvalidOperationException(
                "No observation has been pushed, so there is no lag row to copy.");
        }

        var available = Available;

        if (destination.Length < available)
        {
            throw new ArgumentException(
                $"Destination must hold at least {available} values; {destination.Length} available.",
                nameof(destination));
        }

        // _next points one past the newest value, so lag k sits k + 1 places behind it.
        for (var k = 0; k < available; k++)
        {
            var index = _next - 1 - k;
            destination[k] = _buffer[index < 0 ? index + RowLength : index];
        }

        return available;
    }

    internal void WriteState(BinaryWriter writer)
    {
        writer.Write(RowLength);
        writer.Write(_next);
        writer.Write(_seen);

        foreach (var value in _buffer)
        {
            writer.Write(value);
        }
    }

    internal void ReadState(BinaryReader reader)
    {
        var rowLength = reader.ReadInt32();

        if (rowLength != RowLength)
        {
            throw new InvalidDataException($"Saved lag window has row length {rowLength}; this one has {RowLength}.");
        }

        _next = reader.ReadInt32();
        _seen = reader.ReadInt32();

        for (var i = 0; i < _buffer.Length; i++)
        {
            _buffer[i] = reader.ReadDouble();
        }
    }

    /// <summary>Discards all buffered observations.</summary>
    public void Reset()
    {
        _next = 0;
        _seen = 0;
        Array.Clear(_buffer, 0, _buffer.Length);
    }
}
