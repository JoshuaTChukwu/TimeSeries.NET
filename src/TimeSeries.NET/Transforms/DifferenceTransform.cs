namespace TimeSeries.Transforms;

/// <summary>
/// Applies <c>(1-B)^d (1-B^s)^D</c> to a series in flight, holding only
/// <see cref="DifferenceSpec.WarmupLength"/> values regardless of how long the series is.
/// </summary>
/// <remarks>
/// <para>
/// The operator is composed from single-lag stages rather than special-cased: the
/// seasonal differences run first, then the non-seasonal ones. Because the two operators
/// commute the result is the same either way, but fixing the order fixes the layout of
/// the <see cref="IntegrationState"/> the transform hands back.
/// </para>
/// <para>
/// One instance belongs to one traversal and is not thread-safe. State carries across
/// calls to <see cref="Transform"/>, so a batch boundary does not affect the output.
/// </para>
/// </remarks>
public sealed class DifferenceTransform : ISeriesTransform
{
    private readonly LagDifferenceStage[] _stages;
    private double[] _scratchA = [];
    private double[] _scratchB = [];

    /// <summary>
    /// Creates a transform for the given differencing.
    /// </summary>
    /// <param name="spec">The differencing to apply.</param>
    public DifferenceTransform(DifferenceSpec spec)
    {
        Spec = spec;

        _stages = new LagDifferenceStage[spec.SeasonalOrder + spec.Order];
        var next = 0;

        for (var i = 0; i < spec.SeasonalOrder; i++)
        {
            _stages[next++] = new LagDifferenceStage(spec.Period);
        }

        for (var i = 0; i < spec.Order; i++)
        {
            _stages[next++] = new LagDifferenceStage(1);
        }
    }

    /// <summary>
    /// Creates a transform that continues from where another left off, primed from the
    /// state that transform captured. Its first output corresponds to the observation
    /// immediately after the one the state was captured at.
    /// </summary>
    /// <param name="state">State captured by <see cref="CaptureState"/> or <see cref="IntegrationState.FromEnd"/>.</param>
    /// <returns>A warm transform for <c>state.Spec</c>.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="state"/> is null.</exception>
    public static DifferenceTransform Continue(IntegrationState state)
    {
        if (state is null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        var transform = new DifferenceTransform(state.Spec);
        var tail = state.Tail.Span;
        var offset = 0;

        foreach (var stage in transform._stages)
        {
            stage.Prime(tail.Slice(offset, stage.Lag));
            offset += stage.Lag;
        }

        return transform;
    }

    /// <summary>The differencing this transform applies.</summary>
    public DifferenceSpec Spec { get; }

    /// <inheritdoc />
    public int WarmupLength => Spec.WarmupLength;

    /// <summary>
    /// True once warm-up is complete, meaning every stage has filled its window and
    /// <see cref="CaptureState"/> can be called.
    /// </summary>
    public bool IsWarm
    {
        get
        {
            foreach (var stage in _stages)
            {
                if (!stage.IsWarm)
                {
                    return false;
                }
            }

            return true;
        }
    }

    /// <inheritdoc />
    public int Transform(ReadOnlySpan<double> input, Span<double> output)
    {
        if (output.Length < input.Length)
        {
            throw new ArgumentException(
                $"Output span must be at least as long as the input: {input.Length} values " +
                $"supplied, {output.Length} available.",
                nameof(output));
        }

        if (_stages.Length == 0)
        {
            input.CopyTo(output);
            return input.Length;
        }

        EnsureScratch(input.Length);

        var source = input;
        var count = input.Length;

        for (var i = 0; i < _stages.Length; i++)
        {
            // The final stage writes straight into the caller's buffer. Intermediate
            // stages alternate between two scratch arrays, so a stage never reads the
            // buffer it is writing.
            var destination = i == _stages.Length - 1
                ? output
                : (i % 2 == 0 ? _scratchA : _scratchB).AsSpan(0, input.Length);

            count = _stages[i].Apply(source.Slice(0, count), destination);
            source = destination;
        }

        return count;
    }

    /// <inheritdoc />
    public void Reset()
    {
        foreach (var stage in _stages)
        {
            stage.Reset();
        }
    }

    /// <summary>
    /// Captures the state needed to invert this transform from the current position
    /// forward, without retaining the series.
    /// </summary>
    /// <returns>
    /// An immutable <see cref="IntegrationState"/> holding
    /// <see cref="DifferenceSpec.WarmupLength"/> values.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// Warm-up is not complete, so the stage windows are not yet meaningful.
    /// </exception>
    public IntegrationState CaptureState()
    {
        if (!IsWarm)
        {
            throw new InvalidOperationException(
                $"The transform has not consumed its warm-up of {WarmupLength} observations, " +
                "so there is no integration state to capture.");
        }

        var values = new double[Spec.WarmupLength];
        var offset = 0;

        foreach (var stage in _stages)
        {
            stage.CopyWindowTo(values.AsSpan(offset, stage.Lag));
            offset += stage.Lag;
        }

        return new IntegrationState(Spec, values);
    }

    /// <summary>Writes every stage's state, so a traversal can resume mid-warm-up.</summary>
    internal void WriteState(BinaryWriter writer)
    {
        writer.Write(_stages.Length);

        foreach (var stage in _stages)
        {
            stage.WriteState(writer);
        }
    }

    /// <summary>Restores state written by <see cref="WriteState"/>.</summary>
    internal void ReadState(BinaryReader reader)
    {
        var count = reader.ReadInt32();

        if (count != _stages.Length)
        {
            throw new InvalidDataException($"Saved transform has {count} stages; {Spec} has {_stages.Length}.");
        }

        foreach (var stage in _stages)
        {
            stage.ReadState(reader);
        }
    }

    private void EnsureScratch(int length)
    {
        if (_stages.Length < 2 || _scratchA.Length >= length)
        {
            return;
        }

        _scratchA = new double[length];
        _scratchB = new double[length];
    }
}
