// Public debt, ten years ahead, from views the database owner grants access to.
//
// The library never sees a table. The consumer creates two views —
//   vw_debt_history   (period, debt_gdp, unemployment, inflation, growth, rate)
//   vw_macro_scenario (period, unemployment, inflation, growth, rate)   — the assumed future
// — and grants SELECT on them. The first is fitted; the second supplies the regressor paths
// the forecast needs, because the library will not extrapolate anyone's macro assumptions.
//
// Model: debt/GDP is I(1); its *changes* are driven by the *levels* of the drivers, so the
// regressors enter undifferenced (RegressorDifferencing.None):
//     Δdebt_t = c + φ Δdebt_(t-1) + β₁ unemployment_t + β₂ inflation_t + β₃ growth_t + β₄ rate_t + ε_t
//
// Writes an HTML report to the path given as the first argument, or ./public-debt-report.html.

using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.Data.Sqlite;
using TimeSeries;
using TimeSeries.Data;

var outPath = args.Length > 0 ? args[0] : "public-debt-report.html";
var inv = CultureInfo.InvariantCulture;
var rng = new Random(2027);
double Gaussian() => Math.Sqrt(-2d * Math.Log(1d - rng.NextDouble())) * Math.Cos(2d * Math.PI * rng.NextDouble());

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic history: 1990Q1 .. 2024Q4 (140 quarters), then a 40-quarter scenario
// ─────────────────────────────────────────────────────────────────────────────
const int History = 140;
const int Future = 40;
var start = new DateTime(1990, 1, 1);
string Period(int t) => $"{start.AddMonths(3 * t):yyyy}Q{(start.AddMonths(3 * t).Month - 1) / 3 + 1}";

var total = History + Future;
var unemployment = new double[total];
var inflation = new double[total];
var growth = new double[total];
var rate = new double[total];
double u = 6.0, pi = 3.0, g = 0.6, r = 2.0;

for (var t = 0; t < total; t++)
{
    // Recessions in 2008–09 (t = 72..79) and 2020 (t = 120..123); an inflation burst 2021–23.
    var recession = t is >= 72 and <= 79 ? 3.0 * Math.Exp(-(t - 72) / 4.0) : t is >= 120 and <= 123 ? 5.0 * Math.Exp(-(t - 120) / 1.5) : 0d;
    var burst = t is >= 125 and <= 136 ? 4.0 * Math.Exp(-(t - 125) / 4.0) : 0d;
    u = 6.0 + (0.94 * (u - 6.0)) + (0.25 * Gaussian());
    pi = 2.5 + (0.85 * (pi - 2.5)) + (0.35 * Gaussian());
    g = 0.55 + (0.5 * (g - 0.55)) + (0.35 * Gaussian()) - (0.3 * recession);
    r = (t < 72 ? 1.8 : t < 120 ? 0.8 : 1.3) + (0.9 * (r - (t < 72 ? 1.8 : t < 120 ? 0.8 : 1.3))) + (0.06 * Gaussian());
    unemployment[t] = u + recession;
    inflation[t] = Math.Max(-1, pi + burst);
    growth[t] = g;
    rate[t] = Math.Max(0.05, r);
}

// Debt dynamics in points of GDP per quarter: the change in debt persists, and is pushed
// up by slack and by interest, pulled down by growth and by inflation — a linearised
// version of the debt accumulation identity, calibrated so debt drifts from 55% to
// around 90% of GDP over the history.
var debt = new double[total];
debt[0] = 55;
var change = 0d;
for (var t = 1; t < total; t++)
{
    var push = (0.30 * (unemployment[t] - 6.0)) - (0.12 * (inflation[t] - 2.5)) - (0.50 * (growth[t] - 0.55)) + (0.40 * (rate[t] - 1.2));
    change = (0.35 * change) + 0.15 + push + (0.25 * Gaussian());
    debt[t] = debt[t - 1] + change;
}

// ─────────────────────────────────────────────────────────────────────────────
// The database the consumer owns: tables, then the two views they grant access to
// ─────────────────────────────────────────────────────────────────────────────
var connectionString = "Data Source=file:debt?mode=memory&cache=shared";
using var keepAlive = new SqliteConnection(connectionString);
keepAlive.Open();

void Execute(string sql)
{
    using var command = keepAlive.CreateCommand();
    command.CommandText = sql;
    command.ExecuteNonQuery();
}

Execute("CREATE TABLE fiscal_quarterly (period TEXT PRIMARY KEY, debt_gdp REAL, unemployment REAL, inflation REAL, growth REAL, rate REAL)");
Execute("CREATE TABLE macro_projection (scenario TEXT, period TEXT, unemployment REAL, inflation REAL, growth REAL, rate REAL, PRIMARY KEY (scenario, period))");

using (var tx = keepAlive.BeginTransaction())
{
    using var insert = keepAlive.CreateCommand();
    insert.Transaction = tx;
    insert.CommandText = "INSERT INTO fiscal_quarterly VALUES (@p, @d, @u, @i, @g, @r)";
    var pP = insert.Parameters.Add("@p", SqliteType.Text);
    var pD = insert.Parameters.Add("@d", SqliteType.Real);
    var pU = insert.Parameters.Add("@u", SqliteType.Real);
    var pI = insert.Parameters.Add("@i", SqliteType.Real);
    var pG = insert.Parameters.Add("@g", SqliteType.Real);
    var pR = insert.Parameters.Add("@r", SqliteType.Real);
    for (var t = 0; t < History; t++)
    {
        pP.Value = Period(t); pD.Value = debt[t]; pU.Value = unemployment[t]; pI.Value = inflation[t]; pG.Value = growth[t]; pR.Value = rate[t];
        insert.ExecuteNonQuery();
    }

    using var scenario = keepAlive.CreateCommand();
    scenario.Transaction = tx;
    scenario.CommandText = "INSERT INTO macro_projection VALUES (@s, @p, @u, @i, @g, @r)";
    var sS = scenario.Parameters.Add("@s", SqliteType.Text);
    var sP = scenario.Parameters.Add("@p", SqliteType.Text);
    var sU = scenario.Parameters.Add("@u", SqliteType.Real);
    var sI = scenario.Parameters.Add("@i", SqliteType.Real);
    var sG = scenario.Parameters.Add("@g", SqliteType.Real);
    var sR = scenario.Parameters.Add("@r", SqliteType.Real);
    for (var t = History; t < total; t++)
    {
        // Baseline: the generated path. Adverse: two points more unemployment, one point
        // higher rates, half a point less growth — the kind of stress a supervisor asks for.
        sS.Value = "baseline"; sP.Value = Period(t); sU.Value = unemployment[t]; sI.Value = inflation[t]; sG.Value = growth[t]; sR.Value = rate[t];
        scenario.ExecuteNonQuery();
        sS.Value = "adverse"; sU.Value = unemployment[t] + 2.0; sI.Value = inflation[t]; sG.Value = growth[t] - 0.5; sR.Value = rate[t] + 1.0;
        scenario.ExecuteNonQuery();
    }

    tx.Commit();
}

// The views are the contract. On PostgreSQL or SQL Server this is where the owner runs
//   GRANT SELECT ON vw_debt_history, vw_macro_scenario TO forecasting_role;
// and the forecasting process connects as forecasting_role with nothing else.
Execute("CREATE VIEW vw_debt_history AS SELECT period, debt_gdp, unemployment, inflation, growth, rate FROM fiscal_quarterly");
Execute("CREATE VIEW vw_macro_scenario AS SELECT scenario, period, unemployment, inflation, growth, rate FROM macro_projection");

// ─────────────────────────────────────────────────────────────────────────────
// Fit from the history view
// ─────────────────────────────────────────────────────────────────────────────
string[] drivers = ["unemployment", "inflation", "growth", "rate"];

var history = new DbTimeSeriesSource(
    () => new SqliteConnection(connectionString),
    new DbSeriesQuery
    {
        CommandText = "SELECT period, debt_gdp, unemployment, inflation, growth, rate FROM vw_debt_history ORDER BY period",
        ValueColumn = "debt_gdp",
        ExogenousColumns = drivers,
    });

var auto = new AutoArima(new AutoArimaOptions
{
    MaxP = 2,
    MaxQ = 1,
    MaxPilotOrder = 8,
    Frequency = SeriesFrequency.Quarterly,
    RegressorDifferencing = RegressorDifferencing.None,
});

var selection = await auto.SelectAsync(history);
var fit = selection.Best;

Console.WriteLine("── Public debt / GDP, quarterly, fitted from vw_debt_history ──");
Console.WriteLine($"   observations: {fit.ObservationCount}   KPSS at d=0: {selection.Differencing.First(c => c.Order == 0).Kpss.Statistic:F3} → d={selection.ChosenDifferencing.Order}");
Console.WriteLine($"   selected: ARIMA{fit.Order} with {drivers.Length} level regressors   AICc={fit.Diagnostics.Aicc:F1}");
Console.WriteLine($"   φ=[{string.Join(", ", fit.AutoRegressive.ToArray().Select(v => v.ToString("F3", inv)))}]  θ=[{string.Join(", ", fit.MovingAverage.ToArray().Select(v => v.ToString("F3", inv)))}]  c={fit.Intercept:F3}");
for (var i = 0; i < drivers.Length; i++)
{
    Console.WriteLine($"   β {drivers[i],-13} {fit.ExogenousCoefficients.Span[i],8:F3}   (SE {fit.StandardErrors.Span[1 + fit.Order.P + fit.Order.Q + i]:F3})   → per point, on the quarterly change in debt/GDP");
}

Console.WriteLine($"   Ljung-Box p={fit.Diagnostics.LjungBoxPValue:F3}   stationary={fit.Diagnostics.IsStationary}   invertible={fit.Diagnostics.IsInvertible}");

// ─────────────────────────────────────────────────────────────────────────────
// Forecast ten years under each scenario in the scenario view
// ─────────────────────────────────────────────────────────────────────────────
async Task<(string Name, ExogenousMatrix Path, ForecastResult Forecast)> Scenario(string name)
{
    var source = new DbTimeSeriesSource(
        () => new SqliteConnection(connectionString),
        new DbSeriesQuery
        {
            CommandText = "SELECT period, unemployment, inflation, growth, rate FROM vw_macro_scenario WHERE scenario = @s ORDER BY period",
            ValueColumn = null,                 // regressor-only: nothing to fit, everything to assume
            ExogenousColumns = drivers,
            Parameters = { ["@s"] = name },
        });

    var path = await ExogenousMatrix.FromSourceAsync(source);
    return (name, path, fit.Forecast(ForecastHorizon.Years(10), path));
}

var baseline = await Scenario("baseline");
var adverse = await Scenario("adverse");

Console.WriteLine();
Console.WriteLine("── Ten years ahead, from vw_macro_scenario ──");
Console.WriteLine("   year     baseline        95% interval          adverse     (actual path in the synthetic data)");
for (var h = 3; h < Future; h += 4)
{
    Console.WriteLine($"   {Period(History + h)[..4]}   {baseline.Forecast.Mean.Span[h],7:F1}%   [{baseline.Forecast.Lower.Span[h],6:F1}%, {baseline.Forecast.Upper.Span[h],6:F1}%]   {adverse.Forecast.Mean.Span[h],7:F1}%     {debt[History + h],6:F1}%");
}

foreach (var warning in baseline.Forecast.Warnings.Concat(adverse.Forecast.Warnings).Distinct())
{
    Console.WriteLine($"   WARNING: {warning}");
}

var actualFuture = debt.AsSpan(History, Future).ToArray();
Console.WriteLine();
Console.WriteLine($"   baseline vs the path the data actually followed: MAE {baseline.Forecast.Mean.ToArray().Zip(actualFuture, (f, a) => Math.Abs(f - a)).Average():F2} points of GDP over 40 quarters; " +
                  $"95% interval covered {actualFuture.Zip(baseline.Forecast.Lower.ToArray(), (a, l) => (a, l)).Zip(baseline.Forecast.Upper.ToArray(), (al, up) => al.a >= al.l && al.a <= up).Count(inside => inside) * 100 / Future}% of quarters");
Console.WriteLine($"   interval half-width: {(baseline.Forecast.Upper.Span[0] - baseline.Forecast.Lower.Span[0]) / 2:F1} points at 1 quarter, {(baseline.Forecast.Upper.Span[^1] - baseline.Forecast.Lower.Span[^1]) / 2:F1} at 10 years — the honest content of a decade-ahead number.");

File.WriteAllText(outPath, Report.Build(Enumerable.Range(0, total).Select(Period).ToArray(), debt, History, fit, selection, drivers, baseline.Forecast, adverse.Forecast));
Console.WriteLine();
Console.WriteLine($"report: {Path.GetFullPath(outPath)}");

static class Report
{
    public static string Build(string[] periods, double[] debt, int history, ArimaFit fit, ArimaSelection selection, string[] drivers, ForecastResult baseline, ForecastResult adverse)
    {
        var inv = CultureInfo.InvariantCulture;
        var n = periods.Length;
        var future = n - history;

        const double W = 960, H = 400, Left = 56, Right = 80, Top = 24, Bottom = 44;
        var plotW = W - Left - Right;
        var plotH = H - Top - Bottom;
        var all = debt.Take(history).Concat(baseline.Lower.ToArray()).Concat(baseline.Upper.ToArray()).Concat(adverse.Mean.ToArray()).ToArray();
        var yMax = Math.Ceiling(all.Max() / 10) * 10;
        var yMin = Math.Floor(all.Min() / 10) * 10;
        double X(int t) => Left + (plotW * t / (n - 1));
        double Y(double v) => Top + plotH - (plotH * (v - yMin) / (yMax - yMin));
        string F(double v) => v.ToString("0.##", inv);

        var observed = new StringBuilder();
        for (var t = 0; t < history; t++)
        {
            observed.Append(t == 0 ? "M" : "L").Append(F(X(t))).Append(' ').Append(F(Y(debt[t]))).Append(' ');
        }

        string Line(ReadOnlyMemory<double> values)
        {
            var sb = new StringBuilder($"M{F(X(history - 1))} {F(Y(debt[history - 1]))} ");
            for (var h = 0; h < future; h++)
            {
                sb.Append('L').Append(F(X(history + h))).Append(' ').Append(F(Y(values.Span[h]))).Append(' ');
            }

            return sb.ToString();
        }

        var band = new StringBuilder();
        for (var h = 0; h < future; h++)
        {
            band.Append(h == 0 ? "M" : "L").Append(F(X(history + h))).Append(' ').Append(F(Y(baseline.Upper.Span[h]))).Append(' ');
        }

        for (var h = future - 1; h >= 0; h--)
        {
            band.Append('L').Append(F(X(history + h))).Append(' ').Append(F(Y(baseline.Lower.Span[h]))).Append(' ');
        }

        band.Append('Z');

        var grid = new StringBuilder();
        for (var v = yMin; v <= yMax + 1e-9; v += 10)
        {
            grid.Append($"<line x1=\"{F(Left)}\" x2=\"{F(W - Right)}\" y1=\"{F(Y(v))}\" y2=\"{F(Y(v))}\" class=\"grid\"/>");
            grid.Append($"<text x=\"{F(Left - 8)}\" y=\"{F(Y(v) + 4)}\" class=\"tick\" text-anchor=\"end\">{v.ToString("0", inv)}%</text>");
        }

        var xTicks = new StringBuilder();
        for (var t = 0; t < n; t += 20)
        {
            xTicks.Append($"<text x=\"{F(X(t))}\" y=\"{F(H - 16)}\" class=\"tick\" text-anchor=\"middle\">{periods[t][..4]}</text>");
        }

        var rows = new StringBuilder();
        for (var h = 3; h < future; h += 4)
        {
            rows.Append($"<tr><td>{periods[history + h][..4]}</td><td class=\"num\">{baseline.Mean.Span[h].ToString("F1", inv)}%</td><td class=\"num\">{baseline.Lower.Span[h].ToString("F1", inv)}%</td><td class=\"num\">{baseline.Upper.Span[h].ToString("F1", inv)}%</td><td class=\"num\">{adverse.Mean.Span[h].ToString("F1", inv)}%</td><td class=\"num\">{(adverse.Mean.Span[h] - baseline.Mean.Span[h]).ToString("+0.0;-0.0", inv)}</td></tr>");
        }

        var betas = new StringBuilder();
        for (var i = 0; i < drivers.Length; i++)
        {
            var se = fit.StandardErrors.Span[1 + fit.Order.P + fit.Order.Q + i];
            betas.Append($"<tr><td>{drivers[i]}</td><td class=\"num\">{fit.ExogenousCoefficients.Span[i].ToString("F3", inv)}</td><td class=\"num\">{se.ToString("F3", inv)}</td></tr>");
        }

        var candidates = new StringBuilder();
        foreach (var c in selection.Candidates.Take(6))
        {
            candidates.Append($"<tr><td>ARIMA{c.Order}</td><td class=\"num\">{(c.Succeeded ? c.Aicc.ToString("F1", inv) : "—")}</td><td class=\"num\">{(c.Fit is null ? "—" : c.Fit.Diagnostics.LjungBoxPValue.ToString("F3", inv))}</td><td>{(c.IsAdmissible ? "yes" : "no")}</td></tr>");
        }

        var d = fit.Diagnostics;
        var series = JsonSerializer.Serialize(new
        {
            periods,
            observed = debt.Select((v, t) => t < history ? v : (double?)null),
            baseline = Enumerable.Range(0, n).Select(t => t >= history ? baseline.Mean.Span[t - history] : (double?)null),
            lower = Enumerable.Range(0, n).Select(t => t >= history ? baseline.Lower.Span[t - history] : (double?)null),
            upper = Enumerable.Range(0, n).Select(t => t >= history ? baseline.Upper.Span[t - history] : (double?)null),
            adverse = Enumerable.Range(0, n).Select(t => t >= history ? adverse.Mean.Span[t - history] : (double?)null),
            left = Left, plotW, n,
        });

        return $$"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TimeSeries.NET — public debt, ten years ahead</title>
<style>
  :root { color-scheme: light; --surface-1: #fcfcfb; --plane: #f9f9f7; --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; }
  @media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark; --surface-1: #1a1a19; --plane: #0d0d0d; --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; } }
  :root[data-theme="dark"] { color-scheme: dark; --surface-1: #1a1a19; --plane: #0d0d0d; --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; }
  body { margin: 0; background: var(--plane); color: var(--text-primary); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1040px; margin: 0 auto; padding: 32px 24px 48px; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  h2 { font-size: 16px; font-weight: 600; margin: 32px 0 12px; }
  .sub { color: var(--text-secondary); margin: 0 0 20px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }
  .tile .label { color: var(--text-secondary); font-size: 13px; }
  .tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
  .tile .note { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 16px; position: relative; }
  .legend { display: flex; gap: 20px; flex-wrap: wrap; color: var(--text-secondary); font-size: 13px; margin-bottom: 8px; }
  .legend span { display: inline-flex; align-items: center; gap: 8px; }
  .key { width: 18px; height: 2px; background: var(--series-1); border-radius: 1px; display: inline-block; }
  .key.s2 { background: var(--series-2); } .key.s3 { background: var(--series-3); }
  .key.band { height: 10px; background: var(--series-2); opacity: .18; }
  svg { width: 100%; height: auto; display: block; }
  .grid { stroke: var(--grid); stroke-width: 1; } .axis { stroke: var(--axis); stroke-width: 1; }
  .tick, .lbl { fill: var(--muted); font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; font-variant-numeric: tabular-nums; }
  .lbl { fill: var(--text-secondary); }
  .observed { fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .baseline { fill: none; stroke: var(--series-2); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .adverse { fill: none; stroke: var(--series-3); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .band { fill: var(--series-2); opacity: .12; }
  .end { stroke: var(--surface-1); stroke-width: 2; }
  .crosshair { stroke: var(--axis); stroke-width: 1; display: none; }
  .tooltip { position: absolute; pointer-events: none; display: none; background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.08); min-width: 190px; }
  .tooltip .t { color: var(--text-secondary); margin-bottom: 4px; }
  .tooltip .row { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
  .tooltip .row .v { font-weight: 600; font-variant-numeric: tabular-nums; }
  .tooltip .row .l { color: var(--text-secondary); display: inline-flex; align-items: center; gap: 6px; }
  table { border-collapse: collapse; width: 100%; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  th, td { padding: 7px 12px; text-align: left; border-top: 1px solid var(--grid); }
  th { color: var(--text-secondary); font-weight: 500; border-top: none; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 760px) { .cols { grid-template-columns: 1fr; } }
  code, pre { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--text-secondary); }
  pre { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; overflow-x: auto; }
</style>
</head>
<body>
<main>
  <h1>Public debt — ten years ahead, from two views</h1>
  <p class="sub">Debt/GDP fitted on {{history}} quarters from <code>vw_debt_history</code>; the drivers' assumed paths read from <code>vw_macro_scenario</code>. Δdebt is explained by the <em>levels</em> of unemployment, inflation, growth and the interest rate (<code>RegressorDifferencing.None</code>). The forecast is conditional on each scenario — the library does not extrapolate anyone's macro assumptions.</p>

  <div class="tiles">
    <div class="tile"><div class="label">Selected model</div><div class="value">ARIMA{{fit.Order}}</div><div class="note">AICc {{d.Aicc.ToString("F1", inv)}} · {{drivers.Length}} level regressors · {{selection.Candidates.Count}} candidates on one scan</div></div>
    <div class="tile"><div class="label">Debt/GDP, end of history</div><div class="value">{{debt[history - 1].ToString("F1", inv)}}%</div><div class="note">{{periods[history - 1]}}</div></div>
    <div class="tile"><div class="label">Baseline, ten years out</div><div class="value">{{baseline.Mean.Span[^1].ToString("F1", inv)}}%</div><div class="note">95% interval {{baseline.Lower.Span[^1].ToString("F1", inv)}}% to {{baseline.Upper.Span[^1].ToString("F1", inv)}}% · {{periods[^1]}}</div></div>
    <div class="tile"><div class="label">Adverse, ten years out</div><div class="value">{{adverse.Mean.Span[^1].ToString("F1", inv)}}%</div><div class="note">+2 pts unemployment, +1 pt rates, −0.5 pt growth · {{(adverse.Mean.Span[^1] - baseline.Mean.Span[^1]).ToString("+0.0;-0.0", inv)}} vs baseline</div></div>
    <div class="tile"><div class="label">Ljung–Box p-value</div><div class="value">{{d.LjungBoxPValue.ToString("F3", inv)}}</div><div class="note">Q({{d.LjungBoxLags}}) = {{d.LjungBoxStatistic.ToString("F2", inv)}} · {{(d.RejectsWhiteNoiseResiduals() ? "residuals still autocorrelated" : "residuals look like white noise")}}</div></div>
  </div>

  <div class="card" id="chart-card">
    <div class="legend">
      <span><i class="key"></i>Debt/GDP, observed</span>
      <span><i class="key s2"></i>Baseline scenario</span>
      <span><i class="key band"></i>95% interval (baseline)</span>
      <span><i class="key s3"></i>Adverse scenario</span>
    </div>
    <svg viewBox="0 0 {{F(W)}} {{F(H)}}" role="img" aria-label="Public debt to GDP: observed history and ten-year forecasts under a baseline and an adverse scenario">
      {{grid}}
      <line x1="{{F(Left)}}" x2="{{F(W - Right)}}" y1="{{F(Top + plotH)}}" y2="{{F(Top + plotH)}}" class="axis"/>
      <line x1="{{F(X(history - 1))}}" x2="{{F(X(history - 1))}}" y1="{{F(Top)}}" y2="{{F(Top + plotH)}}" class="axis"/>
      <text x="{{F(X(history - 1) + 6)}}" y="{{F(Top + 12)}}" class="lbl">forecast origin</text>
      <path d="{{band}}" class="band"/>
      <path d="{{observed}}" class="observed"/>
      <path d="{{Line(adverse.Mean)}}" class="adverse"/>
      <path d="{{Line(baseline.Mean)}}" class="baseline"/>
      <circle cx="{{F(X(n - 1))}}" cy="{{F(Y(baseline.Mean.Span[^1]))}}" r="4" class="end" fill="var(--series-2)"/>
      <circle cx="{{F(X(n - 1))}}" cy="{{F(Y(adverse.Mean.Span[^1]))}}" r="4" class="end" fill="var(--series-3)"/>
      <text x="{{F(X(n - 1) + 10)}}" y="{{F(Y(baseline.Mean.Span[^1]) + 4)}}" class="lbl">{{baseline.Mean.Span[^1].ToString("F0", inv)}}%</text>
      <text x="{{F(X(n - 1) + 10)}}" y="{{F(Y(adverse.Mean.Span[^1]) + 4)}}" class="lbl">{{adverse.Mean.Span[^1].ToString("F0", inv)}}%</text>
      {{xTicks}}
      <line id="crosshair" class="crosshair" y1="{{F(Top)}}" y2="{{F(Top + plotH)}}"/>
      <rect id="hit" x="{{F(Left)}}" y="{{F(Top)}}" width="{{F(plotW)}}" height="{{F(plotH)}}" fill="transparent"/>
    </svg>
    <div class="tooltip" id="tooltip"></div>
  </div>

  <div class="cols">
    <div>
      <h2>Year-end levels — table view</h2>
      <table>
        <thead><tr><th>Year</th><th class="num">Baseline</th><th class="num">95% low</th><th class="num">95% high</th><th class="num">Adverse</th><th class="num">Δ</th></tr></thead>
        <tbody>{{rows}}</tbody>
      </table>
    </div>
    <div>
      <h2>Drivers</h2>
      <p class="sub">Coefficient on the quarterly change in debt/GDP per point of the driver's level.</p>
      <table>
        <thead><tr><th>Regressor</th><th class="num">β</th><th class="num">SE</th></tr></thead>
        <tbody>{{betas}}</tbody>
      </table>
      <h2>Order selection</h2>
      <table>
        <thead><tr><th>Candidate</th><th class="num">AICc</th><th class="num">Ljung–Box p</th><th>Admissible</th></tr></thead>
        <tbody>{{candidates}}</tbody>
      </table>
    </div>
  </div>

  <h2>What the database owner grants</h2>
<pre>CREATE VIEW vw_debt_history   AS SELECT period, debt_gdp, unemployment, inflation, growth, rate FROM fiscal_quarterly;
CREATE VIEW vw_macro_scenario AS SELECT scenario, period, unemployment, inflation, growth, rate FROM macro_projection;
GRANT SELECT ON vw_debt_history, vw_macro_scenario TO forecasting_role;   -- nothing else is needed</pre>
</main>
<script type="application/json" id="series">{{series}}</script>
<script>
(function () {
  var data = JSON.parse(document.getElementById('series').textContent);
  var svg = document.querySelector('#chart-card svg'), hit = document.getElementById('hit');
  var cross = document.getElementById('crosshair'), tip = document.getElementById('tooltip'), card = document.getElementById('chart-card');
  function pct(v) { return v == null ? null : v.toFixed(1) + '%'; }
  function add(node) { if (node) tip.appendChild(node); }
  function row(label, value, cls) {
    if (value == null) return null;
    var r = document.createElement('div'); r.className = 'row';
    var l = document.createElement('span'); l.className = 'l';
    var k = document.createElement('i'); k.className = 'key ' + cls; l.appendChild(k); l.appendChild(document.createTextNode(label));
    var v = document.createElement('span'); v.className = 'v'; v.textContent = value;
    r.appendChild(l); r.appendChild(v); return r;
  }
  hit.addEventListener('pointermove', function (evt) {
    var rect = svg.getBoundingClientRect(), scale = rect.width / svg.viewBox.baseVal.width;
    var x = (evt.clientX - rect.left) / scale;
    var t = Math.max(0, Math.min(data.n - 1, Math.round((x - data.left) / data.plotW * (data.n - 1))));
    var cx = data.left + data.plotW * t / (data.n - 1);
    cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.style.display = 'block';
    while (tip.firstChild) tip.removeChild(tip.firstChild);
    var head = document.createElement('div'); head.className = 't'; head.textContent = data.periods[t]; tip.appendChild(head);
    add(row('Observed', pct(data.observed[t]), ''));
    add(row('Baseline', pct(data.baseline[t]), 's2'));
    if (data.lower[t] != null) add(row('95% interval', pct(data.lower[t]) + ' – ' + pct(data.upper[t]), 'band'));
    add(row('Adverse', pct(data.adverse[t]), 's3'));
    tip.style.display = 'block';
    var px = cx * scale + (rect.left - card.getBoundingClientRect().left);
    tip.style.left = (px > rect.width * 0.7 ? px - tip.offsetWidth - 12 : px + 12) + 'px';
    tip.style.top = (rect.top - card.getBoundingClientRect().top + 40) + 'px';
  });
  hit.addEventListener('pointerleave', function () { cross.style.display = 'none'; tip.style.display = 'none'; });
})();
</script>
</body>
</html>
""";
    }
}
