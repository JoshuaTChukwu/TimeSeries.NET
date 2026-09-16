using System.Text.Json;
using TimeSeries.Solvers;
using TimeSeries.Transforms;

namespace TimeSeries.Tests.Forecasting;

/// <summary>
/// Forecasts must match statsmodels for the same model and parameters. The constants in
/// reference/forecast_reference.json were produced by reference/generate_reference_values.py
/// — a reference value nobody can regenerate is a magic number, not a test.
/// </summary>
public class StatsmodelsReferenceTests
{
    private sealed record Case(
        string Name,
        int[] Order,
        int[] Seasonal,
        bool IncludeIntercept,
        double Intercept,
        double[] Phi,
        double[] Theta,
        double[] Beta,
        double Sigma2,
        double[] Series,
        double[][]? Exogenous,
        double[][]? FutureExogenous,
        int Steps,
        double[] ForecastMean,
        double[] ForecastSe);

    private static readonly Lazy<Case[]> Cases = new(() =>
    {
        var path = Path.Combine(AppContext.BaseDirectory, "reference", "forecast_reference.json");
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var options = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };

        return document.RootElement.GetProperty("cases")
            .EnumerateArray()
            .Select(element => element.Deserialize<Case>(options)!)
            .ToArray();
    });

    public static TheoryData<string> Names
    {
        get
        {
            var data = new TheoryData<string>();
            foreach (var c in Cases.Value)
            {
                data.Add(c.Name);
            }

            return data;
        }
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void Forecast_MatchesStatsmodels(string name)
    {
        var c = Cases.Value.Single(x => x.Name == name);
        var fit = Rebuild(c);

        var future = c.FutureExogenous is null
            ? null
            : ExogenousMatrix.FromColumns(Enumerable.Range(0, c.Beta.Length)
                .Select(i => c.FutureExogenous.Select(row => row[i]).ToArray()).ToArray());

        var forecast = future is null
            ? fit.Forecast(ForecastHorizon.Periods(c.Steps))
            : fit.Forecast(ForecastHorizon.Periods(c.Steps), future);

        for (var h = 0; h < c.Steps; h++)
        {
            Assert.True(
                Math.Abs(forecast.Mean.Span[h] - c.ForecastMean[h]) < 1e-6,
                $"{name}: mean at step {h + 1} is {forecast.Mean.Span[h]}, statsmodels says {c.ForecastMean[h]}.");
            Assert.True(
                Math.Abs(forecast.StandardError.Span[h] - c.ForecastSe[h]) < 1e-6 * Math.Max(1d, c.ForecastSe[h]),
                $"{name}: standard error at step {h + 1} is {forecast.StandardError.Span[h]}, statsmodels says {c.ForecastSe[h]}.");
        }
    }

    [Fact]
    public void ReferenceFile_CoversTheModelFamilies()
    {
        var names = Cases.Value.Select(c => c.Name).ToArray();

        Assert.Contains("arima010", names);
        Assert.Contains("sarima110_010_12", names);
        Assert.Contains("arimax011", names);
        Assert.True(names.Length >= 12);
    }

    /// <summary>
    /// Assembles a fit from the reference parameters. The seed is computed here from the
    /// textbook recursion, independently of the library's estimator, so the comparison
    /// isolates the forecasting arithmetic.
    /// </summary>
    private static ArimaFit Rebuild(Case c)
    {
        var options = new ArimaOptions
        {
            Order = new(c.Order[0], c.Order[1], c.Order[2]),
            Seasonal = new(c.Seasonal[0], c.Seasonal[1]),
            IncludeIntercept = c.IncludeIntercept,
        };
        var spec = options.Differencing;
        var p = c.Phi.Length;
        var q = c.Theta.Length;
        var r = c.Beta.Length;

        var z = Differencing.Difference(c.Series, spec);
        var x = new double[r][];
        var regressorStates = new IntegrationState[r];

        for (var i = 0; i < r; i++)
        {
            var column = c.Exogenous!.Select(row => row[i]).ToArray();
            x[i] = Differencing.Difference(column, spec);
            regressorStates[i] = IntegrationState.FromEnd(column, spec);
        }

        // e_t = z_t - c - sum phi z_(t-k) - sum theta e_(t-j) - beta' x_t, from t = p with
        // earlier innovations zero. The influence of that start decays as the MA roots.
        var e = new double[z.Length];
        for (var t = p; t < z.Length; t++)
        {
            var value = z[t] - c.Intercept;
            for (var k = 1; k <= p; k++)
            {
                value -= c.Phi[k - 1] * z[t - k];
            }

            for (var j = 1; j <= q; j++)
            {
                if (t - j >= 0)
                {
                    value -= c.Theta[j - 1] * e[t - j];
                }
            }

            for (var i = 0; i < r; i++)
            {
                value -= c.Beta[i] * x[i][t];
            }

            e[t] = value;
        }

        var seed = new ForecastSeed(
            IntegrationState.FromEnd(c.Series, spec),
            regressorStates,
            z.Skip(z.Length - p).ToArray(),
            e.Skip(e.Length - q).ToArray());

        var diagnostics = new ArimaDiagnostics(
            z.Length, p + q + r + 2, 0, 0, 0, 0, 0, 0,
            isStationary: true, stationarityMargin: 1, isInvertible: true, invertibilityMargin: 1,
            isConstantSeries: false, new SolveDiagnostics { Succeeded = true, FailedColumn = -1 },
            0, double.NaN, 0, double.NaN, []);

        return new ArimaFit(
            options, c.Phi, c.Theta, c.Beta, c.Intercept, c.Sigma2, c.Series.Length,
            diagnostics, new FitWindow(0, c.Series.Length), [], seed);
    }
}
