using System.Text;
using TimeSeries.Solvers;
using TimeSeries.Transforms;

namespace TimeSeries;

/// <content>Binary serialisation. A fit is a few hundred bytes and holds no data, so it can be cached, shipped, and forecast from anywhere.</content>
public sealed partial class ArimaFit
{
    private const string Magic = "TSAF";
    private const int FormatVersion = 1;

    /// <summary>
    /// Writes the fit — coefficients, diagnostics, window and the bounded forecasting
    /// tail — in a compact binary form.
    /// </summary>
    /// <param name="destination">A writable stream. Left open.</param>
    /// <exception cref="ArgumentNullException"><paramref name="destination"/> is null.</exception>
    public void WriteTo(Stream destination)
    {
        if (destination is null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        writer.Write(Magic);
        writer.Write(FormatVersion);

        // Options
        writer.Write(Options.Order.P);
        writer.Write(Options.Order.D);
        writer.Write(Options.Order.Q);
        writer.Write(Options.Seasonal.D);
        writer.Write(Options.Seasonal.Period);
        writer.Write(Options.IncludeIntercept);
        writer.Write(Options.PilotOrder ?? -1);
        writer.Write(Options.MaxPilotOrder);
        writer.Write(Options.Ridge);
        writer.Write(Options.ForgettingFactor);
        writer.Write(Options.Frequency.HasValue ? (int)Options.Frequency.Value : 0);

        // Coefficients
        WriteArray(writer, _phi);
        WriteArray(writer, _theta);
        WriteArray(writer, _beta);
        writer.Write(Intercept);
        writer.Write(InnovationVariance);
        writer.Write(ObservationCount);
        WriteArray(writer, _standardErrors);

        // Window
        writer.Write(Window.From);
        writer.Write(Window.To);

        // Diagnostics
        var d = Diagnostics;
        writer.Write(d.EffectiveObservations);
        writer.Write(d.ParameterCount);
        writer.Write(d.ResidualSumOfSquares);
        writer.Write(d.LogLikelihood);
        writer.Write(d.Aic);
        writer.Write(d.Aicc);
        writer.Write(d.Bic);
        writer.Write(d.PilotOrder);
        writer.Write(d.IsStationary);
        writer.Write(d.StationarityMargin);
        writer.Write(d.IsInvertible);
        writer.Write(d.InvertibilityMargin);
        writer.Write(d.IsConstantSeries);
        writer.Write(d.Solve.Dimension);
        writer.Write(d.Solve.Succeeded);
        writer.Write(d.Solve.SmallestPivot);
        writer.Write(d.Solve.LargestPivot);
        writer.Write(d.Solve.Ridge);
        writer.Write(d.Solve.ColumnsScaled);
        writer.Write(d.Solve.FailedColumn);

        // Forecast seed
        WriteArray(writer, Seed.Integration.Tail.ToArray());
        writer.Write(Seed.RegressorStates.Length);
        foreach (var state in Seed.RegressorStates)
        {
            WriteArray(writer, state.Tail.ToArray());
        }

        WriteArray(writer, Seed.RecentValues);
        WriteArray(writer, Seed.RecentResiduals);
        writer.Flush();
    }

    /// <summary>
    /// Reads a fit written by <see cref="WriteTo"/>.
    /// </summary>
    /// <param name="source">The stream to read. Left open.</param>
    /// <returns>The fit, forecasting exactly as the original did.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    /// <exception cref="InvalidDataException">The stream does not hold a fit this library can read.</exception>
    public static ArimaFit ReadFrom(Stream source)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);

        if (!string.Equals(reader.ReadString(), Magic, StringComparison.Ordinal))
        {
            throw new InvalidDataException("The stream does not hold a serialised ArimaFit.");
        }

        var version = reader.ReadInt32();

        if (version != FormatVersion)
        {
            throw new InvalidDataException($"Serialised fit is format version {version}; this library reads version {FormatVersion}.");
        }

        var p = reader.ReadInt32();
        var dOrder = reader.ReadInt32();
        var q = reader.ReadInt32();
        var seasonalD = reader.ReadInt32();
        var period = reader.ReadInt32();
        var intercept = reader.ReadBoolean();
        var pilot = reader.ReadInt32();
        var maxPilot = reader.ReadInt32();
        var ridge = reader.ReadDouble();
        var forgetting = reader.ReadDouble();
        var frequency = reader.ReadInt32();

        var options = new ArimaOptions
        {
            Order = new(p, dOrder, q),
            Seasonal = new(seasonalD, period),
            IncludeIntercept = intercept,
            PilotOrder = pilot < 0 ? null : pilot,
            MaxPilotOrder = maxPilot,
            Ridge = ridge,
            ForgettingFactor = forgetting,
            Frequency = frequency == 0 ? null : (SeriesFrequency)frequency,
        };

        var phi = ReadArray(reader);
        var theta = ReadArray(reader);
        var beta = ReadArray(reader);
        var constant = reader.ReadDouble();
        var sigma2 = reader.ReadDouble();
        var observations = reader.ReadInt64();
        var standardErrors = ReadArray(reader);

        var window = new FitWindow(reader.ReadInt64(), reader.ReadInt64());

        var effective = reader.ReadDouble();
        var parameters = reader.ReadInt32();
        var rss = reader.ReadDouble();
        var logLikelihood = reader.ReadDouble();
        var aic = reader.ReadDouble();
        var aicc = reader.ReadDouble();
        var bic = reader.ReadDouble();
        var pilotOrder = reader.ReadInt32();
        var stationary = reader.ReadBoolean();
        var stationarityMargin = reader.ReadDouble();
        var invertible = reader.ReadBoolean();
        var invertibilityMargin = reader.ReadDouble();
        var constantSeries = reader.ReadBoolean();
        var solve = new SolveDiagnostics
        {
            Dimension = reader.ReadInt32(),
            Succeeded = reader.ReadBoolean(),
            SmallestPivot = reader.ReadDouble(),
            LargestPivot = reader.ReadDouble(),
            Ridge = reader.ReadDouble(),
            ColumnsScaled = reader.ReadBoolean(),
            FailedColumn = reader.ReadInt32(),
        };

        var diagnostics = new ArimaDiagnostics(
            effective, parameters, rss, logLikelihood, aic, aicc, bic, pilotOrder,
            stationary, stationarityMargin, invertible, invertibilityMargin, constantSeries, solve);

        var spec = options.Differencing;
        var integration = new IntegrationState(spec, ReadArray(reader));
        var regressorStates = new IntegrationState[reader.ReadInt32()];
        for (var i = 0; i < regressorStates.Length; i++)
        {
            regressorStates[i] = new IntegrationState(spec, ReadArray(reader));
        }

        var recentValues = ReadArray(reader);
        var recentResiduals = ReadArray(reader);

        if (phi.Length != p || theta.Length != q || beta.Length != regressorStates.Length
            || recentValues.Length != p || recentResiduals.Length != q)
        {
            throw new InvalidDataException("Serialised fit is internally inconsistent.");
        }

        var seed = new ForecastSeed(integration, regressorStates, recentValues, recentResiduals);

        return new ArimaFit(
            options, phi, theta, beta, constant, sigma2, observations, diagnostics, window, standardErrors, seed);
    }

    private static void WriteArray(BinaryWriter writer, double[] values)
    {
        writer.Write(values.Length);

        foreach (var value in values)
        {
            writer.Write(value);
        }
    }

    private static double[] ReadArray(BinaryReader reader)
    {
        var length = reader.ReadInt32();

        if (length < 0 || length > 1 << 20)
        {
            throw new InvalidDataException($"Serialised array length {length} is not plausible.");
        }

        var values = new double[length];

        for (var i = 0; i < length; i++)
        {
            values[i] = reader.ReadDouble();
        }

        return values;
    }
}
