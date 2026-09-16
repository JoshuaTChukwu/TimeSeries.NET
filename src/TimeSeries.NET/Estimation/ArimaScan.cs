using TimeSeries.Accumulators;
using TimeSeries.Solvers;
using TimeSeries.Transforms;

namespace TimeSeries;

/// <summary>
/// The single estimation path. Accepts the series in batches — from an array or from a
/// cursor, it cannot tell — differences in flight, feeds one lag window into the
/// accumulators, and solves once the data is gone. Nothing proportional to the series
/// length is retained.
/// </summary>
internal sealed class ArimaScan
{
    /// <summary>
    /// Rows kept past the lag depth so the final-model innovations can be reconstructed
    /// by recursion at solve time. The recursion converges geometrically for an invertible
    /// MA part, so this many rows leaves the truncation error far below double precision.
    /// </summary>
    internal const int ResidualWarmup = 128;

    private readonly ArimaOptions _options;
    private readonly int _p;
    private readonly int _q;
    private readonly int _r;
    private readonly GramLayout _layout;
    private readonly DifferenceTransform _yTransform;
    private readonly DifferenceTransform[] _xTransforms;
    private readonly LagWindow _yWindow;
    private readonly LagWindow[] _xWindows;
    private readonly LagGramAccumulator _gram;
    private readonly MomentsAccumulator _moments;
    private readonly KpssAccumulator? _kpss;
    private readonly LagWindow? _stationarityWindow;
    private readonly double[] _stationarityRow;
    private readonly double[] _row;
    private readonly double[] _lagScratch;
    private readonly double[] _cx;
    private readonly double[] _tail;
    private readonly int _tailCapacity;
    private readonly int _tailStride;
    private double _cz;
    private bool _offsetsCaptured;
    private int _tailNext;
    private int _tailCount;
    private long _rawCount;
    private long _differencedCount;
    private double[] _zScratch = [];
    private double[] _columnScratch = [];
    private double[][] _xScratch = [];

    internal ArimaScan(ArimaOptions options, int regressorCount, int stationarityLags = 0)
    {
        options.Validate();

        if (stationarityLags > 0)
        {
            _kpss = new KpssAccumulator(stationarityLags);
            _stationarityWindow = new LagWindow(stationarityLags);
        }

        _stationarityRow = new double[stationarityLags + 1];

        _options = options;
        _p = options.Order.P;
        _q = options.Order.Q;
        _r = regressorCount;
        _layout = new GramLayout(options.GramDepth, regressorCount, options.ExogenousLagDepth);

        var spec = options.Differencing;
        _yTransform = new DifferenceTransform(spec);
        _xTransforms = new DifferenceTransform[regressorCount];
        _xWindows = new LagWindow[regressorCount];

        for (var i = 0; i < regressorCount; i++)
        {
            _xTransforms[i] = new DifferenceTransform(spec);
            _xWindows[i] = new LagWindow(_layout.ExogenousLags);
        }

        _yWindow = new LagWindow(_layout.LagDepth);
        _gram = new LagGramAccumulator(_layout.Dimension - 1);
        _moments = new MomentsAccumulator();
        _row = new double[_layout.Dimension];
        _lagScratch = new double[_layout.ExogenousLags + 1];
        _cx = new double[regressorCount];

        _tailStride = 1 + regressorCount;
        _tailCapacity = _layout.LagDepth + 1 + ResidualWarmup;
        _tail = new double[_tailCapacity * _tailStride];
    }

    internal long RawCount => _rawCount;

    internal long DifferencedCount => _differencedCount;

    internal int LagDepth => _layout.LagDepth;

    /// <summary>The Gram accumulator, exposed for the batch-invariance and merge tests.</summary>
    internal LagGramAccumulator Gram => _gram;

    /// <summary>The KPSS result over the differenced series, when tracking was enabled.</summary>
    internal KpssResult? Stationarity => _kpss?.Freeze();

    /// <summary>The moments of the differenced series so far.</summary>
    internal Moments Moments => _moments.Freeze();

    /// <summary>
    /// Folds one batch. Batches may be any length; a boundary is invisible to the result.
    /// </summary>
    /// <param name="values">Raw series values, in time order.</param>
    /// <param name="exogenous">
    /// <c>values.Length * r</c> regressor values, row-major, or empty when the model has none.
    /// </param>
    internal void Accept(ReadOnlySpan<double> values, ReadOnlySpan<double> exogenous)
    {
        var count = values.Length;

        if (exogenous.Length != count * _r)
        {
            throw new ArgumentException(
                $"{count} observations with {_r} regressors need {count * _r} exogenous values; " +
                $"{exogenous.Length} supplied.",
                nameof(exogenous));
        }

        SeriesGuard.Finite(values, "series", _rawCount);
        SeriesGuard.Finite(exogenous, "exogenous", _rawCount * _r);
        EnsureScratch(count);

        var produced = _yTransform.Transform(values, _zScratch);

        for (var i = 0; i < _r; i++)
        {
            for (var t = 0; t < count; t++)
            {
                _columnScratch[t] = exogenous[(t * _r) + i];
            }

            var producedX = _xTransforms[i].Transform(_columnScratch.AsSpan(0, count), _xScratch[i]);

            if (producedX != produced)
            {
                throw new InvalidOperationException(
                    "Internal error: the series and a regressor differenced to different lengths.");
            }
        }

        for (var t = 0; t < produced; t++)
        {
            var z = _zScratch[t];

            // Per the numerical contract, every column is shifted by its first differenced
            // value so that sums of squares measure deviations rather than level. The
            // intercept absorbs the shift at solve time; without an intercept there is
            // nothing to absorb it, so nothing is shifted.
            if (!_offsetsCaptured)
            {
                if (_options.IncludeIntercept)
                {
                    _cz = z;

                    for (var i = 0; i < _r; i++)
                    {
                        _cx[i] = _xScratch[i][t];
                    }
                }

                _offsetsCaptured = true;
            }

            var shifted = z - _cz;
            _yWindow.Push(shifted);
            _moments.Add(shifted);

            if (_kpss is not null)
            {
                _stationarityWindow!.Push(shifted);
                var available = _stationarityWindow.CopyRow(_stationarityRow);
                _kpss.Add(_stationarityRow.AsSpan(0, available));
            }

            var tailBase = _tailNext * _tailStride;
            _tail[tailBase] = shifted;

            for (var i = 0; i < _r; i++)
            {
                var x = _xScratch[i][t] - _cx[i];
                _xWindows[i].Push(x);
                _tail[tailBase + 1 + i] = x;
            }

            _tailNext = _tailNext + 1 == _tailCapacity ? 0 : _tailNext + 1;

            if (_tailCount < _tailCapacity)
            {
                _tailCount++;
            }

            if (_yWindow.IsFull)
            {
                _yWindow.CopyRow(_row.AsSpan(0, _layout.LagDepth + 1));

                for (var i = 0; i < _r; i++)
                {
                    // The window hands back lags 0..q of one regressor contiguously; the
                    // row keeps them lag-major so a shift by j is a block move.
                    _xWindows[i].CopyRow(_lagScratch);

                    for (var lag = 0; lag <= _layout.ExogenousLags; lag++)
                    {
                        _row[_layout.Exogenous(lag, i)] = _lagScratch[lag];
                    }
                }

                _gram.Add(_row);
            }

            _differencedCount++;
        }

        _rawCount += count;
    }

    /// <summary>
    /// Ages the accumulated state by the forgetting factor: <c>G</c>, <c>s</c> and
    /// <c>N</c> all scale by <paramref name="lambda"/>, so older observations count for
    /// less. The tail and the transforms are untouched — they describe where the series
    /// is, not how much of it there was.
    /// </summary>
    internal void Scale(double lambda)
    {
        _gram.Scale(lambda);
        _moments.Scale(lambda);
    }

    private const int StateVersion = 2;

    /// <summary>
    /// Writes the complete scan state — transforms, windows, accumulators, offsets, tail
    /// and counts — so a fold can resume later, elsewhere, exactly.
    /// </summary>
    internal void WriteState(BinaryWriter writer)
    {
        if (_kpss is not null)
        {
            throw new NotSupportedException("A scan tracking stationarity for order selection cannot be saved.");
        }

        writer.Write(StateVersion);
        writer.Write(_p);
        writer.Write(_options.Order.D);
        writer.Write(_q);
        writer.Write(_options.Seasonal.D);
        writer.Write(_options.Seasonal.Period);
        writer.Write(_options.IncludeIntercept);
        writer.Write(_layout.LagDepth);
        writer.Write(_layout.ExogenousLags);
        writer.Write(_r);

        _yTransform.WriteState(writer);
        foreach (var transform in _xTransforms)
        {
            transform.WriteState(writer);
        }

        _yWindow.WriteState(writer);
        foreach (var window in _xWindows)
        {
            window.WriteState(writer);
        }

        _gram.WriteState(writer);
        _moments.WriteState(writer);

        writer.Write(_offsetsCaptured);
        writer.Write(_cz);
        foreach (var offset in _cx)
        {
            writer.Write(offset);
        }

        writer.Write(_tailCapacity);
        writer.Write(_tailNext);
        writer.Write(_tailCount);
        foreach (var value in _tail)
        {
            writer.Write(value);
        }

        writer.Write(_rawCount);
        writer.Write(_differencedCount);
    }

    /// <summary>Restores state written by <see cref="WriteState"/> into a freshly built scan.</summary>
    internal void ReadState(BinaryReader reader)
    {
        var version = reader.ReadInt32();

        if (version != StateVersion)
        {
            throw new InvalidDataException($"Saved scan state is version {version}; this library reads version {StateVersion}.");
        }

        var p = reader.ReadInt32();
        var d = reader.ReadInt32();
        var q = reader.ReadInt32();
        var seasonalD = reader.ReadInt32();
        var period = reader.ReadInt32();
        var intercept = reader.ReadBoolean();
        var lagDepth = reader.ReadInt32();
        var exogenousLags = reader.ReadInt32();
        var r = reader.ReadInt32();

        if (p != _p || d != _options.Order.D || q != _q || seasonalD != _options.Seasonal.D
            || period != _options.Seasonal.Period || intercept != _options.IncludeIntercept
            || lagDepth != _layout.LagDepth || exogenousLags != _layout.ExogenousLags || r != _r)
        {
            throw new InvalidDataException(
                $"Saved state is for ARIMA({p},{d},{q}) seasonal ({seasonalD})[{period}], intercept {intercept}, " +
                $"lag depth {lagDepth}, {r} regressor(s); this estimator is ARIMA{_options.Order} seasonal " +
                $"{_options.Seasonal}, intercept {_options.IncludeIntercept}, lag depth {_layout.LagDepth}, {_r} regressor(s).");
        }

        _yTransform.ReadState(reader);
        foreach (var transform in _xTransforms)
        {
            transform.ReadState(reader);
        }

        _yWindow.ReadState(reader);
        foreach (var window in _xWindows)
        {
            window.ReadState(reader);
        }

        _gram.ReadState(reader);
        _moments.ReadState(reader);

        _offsetsCaptured = reader.ReadBoolean();
        _cz = reader.ReadDouble();
        for (var i = 0; i < _cx.Length; i++)
        {
            _cx[i] = reader.ReadDouble();
        }

        var capacity = reader.ReadInt32();

        if (capacity != _tailCapacity)
        {
            throw new InvalidDataException($"Saved tail capacity {capacity} differs from {_tailCapacity}.");
        }

        _tailNext = reader.ReadInt32();
        _tailCount = reader.ReadInt32();
        for (var i = 0; i < _tail.Length; i++)
        {
            _tail[i] = reader.ReadDouble();
        }

        _rawCount = reader.ReadInt64();
        _differencedCount = reader.ReadInt64();
    }

    /// <summary>Solves the scan's own model from the accumulated state. Touches no data.</summary>
    internal ArimaFit Solve(FitWindow window) => Solve(_options, window);

    /// <summary>
    /// Solves a candidate model from the accumulated state. Any <c>(p, q)</c> no larger
    /// than the scan's own is a sub-problem of the same matrix, which is what makes an
    /// order search free after one pass: every candidate uses the same rows, so their
    /// information criteria are comparable.
    /// </summary>
    /// <exception cref="ArgumentException">The candidate is not a sub-problem of this scan.</exception>
    internal ArimaFit Solve(ArimaOptions candidate, FitWindow window)
    {
        candidate.Validate();

        if (candidate.Order.P > _p || candidate.Order.Q > _q
            || candidate.Differencing != _options.Differencing
            || candidate.IncludeIntercept != _options.IncludeIntercept
            || candidate.LjungBoxLags != _options.LjungBoxLags
            || candidate.GramDepth > _layout.LagDepth)
        {
            throw new ArgumentException(
                $"ARIMA{candidate.Order} is not a sub-problem of the scan built for ARIMA{_options.Order} " +
                $"(lag depth {_layout.LagDepth}, differencing {_options.Differencing}).",
                nameof(candidate));
        }

        var p = candidate.Order.P;
        var q = candidate.Order.Q;
        var lagDepth = _layout.LagDepth;
        var required = lagDepth + (10L * (p + q + _r + 1));

        if (_differencedCount < required)
        {
            var spec = candidate.Differencing;
            throw new InsufficientDataException(
                required, _differencedCount,
                $"ARIMA{candidate.Order}{(spec.IsIdentity ? string.Empty : " with " + spec)} needs at least " +
                $"{required} observations after differencing — the lag depth {lagDepth} plus ten per " +
                $"coefficient — and {_differencedCount} are available from {_rawCount} raw rows. " +
                "Supply a longer series, or a lower order or MaxPilotOrder.");
        }

        var gram = _gram.Freeze();
        var moments = _moments.Freeze();
        var integration = _yTransform.CaptureState();
        var regressorStates = new IntegrationState[_r];

        for (var i = 0; i < _r; i++)
        {
            regressorStates[i] = _xTransforms[i].CaptureState();
        }

        if (moments.Maximum.Equals(moments.Minimum))
        {
            return ConstantFit(candidate, moments, integration, regressorStates, window);
        }

        var solver = new NormalEquationSolver(ridge: candidate.Ridge);
        var hr = HannanRissanen.Solve(gram, _layout, candidate, solver);

        // Un-shift the intercept: z - cz = c~ + sum phi (z - cz) + beta'(x - cx) + ...
        var intercept = hr.Intercept;

        if (_options.IncludeIntercept)
        {
            var phiSum = 0d;
            foreach (var phi in hr.Phi)
            {
                phiSum += phi;
            }

            intercept += _cz * (1d - phiSum);

            for (var i = 0; i < _r; i++)
            {
                intercept -= hr.Beta[i] * _cx[i];
            }
        }

        var n = gram.Count;
        var sigma2 = hr.ResidualSumOfSquares / n;
        var k = hr.ParameterCount + 1;
        var logLikelihood = -0.5 * n * (Math.Log(2d * Math.PI * sigma2) + 1d);
        var aic = (-2d * logLikelihood) + (2d * k);
        var aicc = aic + (2d * k * (k + 1) / (n - k - 1));
        var bic = (-2d * logLikelihood) + (k * Math.Log(n));

        var stationary = PolynomialStability.IsStationary(hr.Phi, out var stationarityMargin);
        var invertible = PolynomialStability.IsInvertible(hr.Theta, out var invertibilityMargin);

        // Ljung-Box on the second-stage residuals, from their exact autocovariances.
        var lags = candidate.LjungBoxLags;
        var residualAcf = new double[lags];
        var ljungBox = double.NaN;
        var ljungBoxDf = lags - p - q;
        var ljungBoxP = double.NaN;

        if (lags > 0 && hr.ResidualAutocovariances[0] > 0d)
        {
            var statistic = 0d;

            for (var lag = 1; lag <= lags; lag++)
            {
                residualAcf[lag - 1] = hr.ResidualAutocovariances[lag] / hr.ResidualAutocovariances[0];
                statistic += residualAcf[lag - 1] * residualAcf[lag - 1] / (n - lag);
            }

            ljungBox = n * (n + 2d) * statistic;
            ljungBoxP = ljungBoxDf > 0 ? Statistics.ChiSquaredDistribution.SurvivalFunction(ljungBox, ljungBoxDf) : double.NaN;
        }

        var diagnostics = new ArimaDiagnostics(
            n, k, hr.ResidualSumOfSquares, logLikelihood, aic, aicc, bic, hr.PilotOrder,
            stationary, stationarityMargin, invertible, invertibilityMargin,
            isConstantSeries: false, hr.Solve,
            lags, ljungBox, ljungBoxDf, ljungBoxP, residualAcf);

        var seed = new ForecastSeed(integration, regressorStates, RecentValues(p), RecentResiduals(hr, p, q));

        return new ArimaFit(
            candidate, hr.Phi, hr.Theta, hr.Beta, intercept, sigma2, _rawCount,
            diagnostics, window, hr.StandardErrors, seed);
    }

    /// <summary>
    /// The standing decision: an exactly constant differenced series is a deterministic
    /// model, not a singular regression.
    /// </summary>
    private ArimaFit ConstantFit(
        ArimaOptions candidate, Moments moments, IntegrationState integration, IntegrationState[] regressorStates, FitWindow window)
    {
        var p = candidate.Order.P;
        var q = candidate.Order.Q;
        var intercept = _cz + moments.Minimum;
        var count = (candidate.IncludeIntercept ? 1 : 0) + p + q + _r;
        var standardErrors = new double[count];

        for (var i = 0; i < count; i++)
        {
            standardErrors[i] = double.NaN;
        }

        var diagnostics = new ArimaDiagnostics(
            _gram.Count, count + 1, 0d, double.NaN, double.NaN, double.NaN, double.NaN, 0,
            isStationary: true, stationarityMargin: 1d, isInvertible: true, invertibilityMargin: 1d,
            isConstantSeries: true,
            new SolveDiagnostics { Dimension = count, Succeeded = true, FailedColumn = -1 },
            candidate.LjungBoxLags, double.NaN, candidate.LjungBoxLags - p - q, double.NaN, new double[candidate.LjungBoxLags]);

        var seed = new ForecastSeed(integration, regressorStates, RecentValues(p), new double[q]);

        return new ArimaFit(
            candidate, new double[p], new double[q], new double[_r], intercept, 0d, _rawCount,
            diagnostics, window, standardErrors, seed);
    }

    private double TailValue(int chronologicalIndex)
    {
        var start = _tailCount < _tailCapacity ? 0 : _tailNext;
        var physical = start + chronologicalIndex;

        if (physical >= _tailCapacity)
        {
            physical -= _tailCapacity;
        }

        return _tail[physical * _tailStride];
    }

    private double TailRegressor(int chronologicalIndex, int regressor)
    {
        var start = _tailCount < _tailCapacity ? 0 : _tailNext;
        var physical = start + chronologicalIndex;

        if (physical >= _tailCapacity)
        {
            physical -= _tailCapacity;
        }

        return _tail[(physical * _tailStride) + 1 + regressor];
    }

    /// <summary>The last <c>p</c> differenced values, back on the unshifted differenced scale.</summary>
    private double[] RecentValues(int p)
    {
        var recent = new double[p];
        var take = Math.Min(p, _tailCount);

        for (var i = 0; i < take; i++)
        {
            recent[p - take + i] = TailValue(_tailCount - take + i) + _cz;
        }

        return recent;
    }

    /// <summary>
    /// The last <c>q</c> innovations of the fitted model, reconstructed over the retained
    /// tail: pilot residuals seed the recursion, and the fitted ARMA recursion then runs
    /// forward far enough for the seeding to have washed out.
    /// </summary>
    private double[] RecentResiduals(HannanRissanen.Result hr, int p, int q)
    {
        var residuals = new double[q];

        if (q == 0)
        {
            return residuals;
        }

        var m = hr.PilotOrder;
        var length = _tailCount;
        var start = m + q;

        if (length <= start)
        {
            return residuals;
        }

        var pilot = new double[length];

        for (var t = m; t < length; t++)
        {
            var value = TailValue(t) - hr.PilotIntercept;

            for (var k = 1; k <= m; k++)
            {
                value -= hr.PilotPhi[k - 1] * TailValue(t - k);
            }

            for (var i = 0; i < _r; i++)
            {
                value -= hr.PilotBeta[i] * TailRegressor(t, i);
            }

            pilot[t] = value;
        }

        var final = new double[length];

        for (var t = start; t < length; t++)
        {
            var value = TailValue(t) - hr.Intercept;

            for (var k = 1; k <= p; k++)
            {
                value -= hr.Phi[k - 1] * TailValue(t - k);
            }

            for (var j = 1; j <= q; j++)
            {
                var previous = t - j >= start ? final[t - j] : pilot[t - j];
                value -= hr.Theta[j - 1] * previous;
            }

            for (var i = 0; i < _r; i++)
            {
                value -= hr.Beta[i] * TailRegressor(t, i);
            }

            final[t] = value;
        }

        Array.Copy(final, length - q, residuals, 0, q);
        return residuals;
    }

    private void EnsureScratch(int count)
    {
        if (_zScratch.Length >= count)
        {
            return;
        }

        _zScratch = new double[count];
        _columnScratch = new double[count];
        _xScratch = new double[_r][];

        for (var i = 0; i < _r; i++)
        {
            _xScratch[i] = new double[count];
        }
    }
}
