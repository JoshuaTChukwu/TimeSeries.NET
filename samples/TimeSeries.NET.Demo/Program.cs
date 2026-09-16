// TimeSeries.NET demo: what the library looks like on data shaped like a lender's.
//
//   1. An aggregate monthly default rate with an unemployment regressor, modelled on the
//      probit scale, the last 24 months held out. AutoArima picks the order; the forecast
//      is compared with what actually happened.
//   2. A population of 300 accounts read from SQLite in one grouped scan, fitted in
//      parallel, compared with the sequential path.
//   3. The incremental estimator: save state, fold the held-out months, solve.
//
// Writes an HTML report (chart, tooltips, table view) to the path given as the first
// argument, or ./demo-report.html.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.Data.Sqlite;
using TimeSeries;
using TimeSeries.Data;
using TimeSeries.Statistics;

var outPath = args.Length > 0 ? args[0] : "demo-report.html";
var inv = CultureInfo.InvariantCulture;
var rng = new Random(2026);
double Gaussian() => Math.Sqrt(-2d * Math.Log(1d - rng.NextDouble())) * Math.Cos(2d * Math.PI * rng.NextDouble());

// ─────────────────────────────────────────────────────────────────────────────
// 1. Aggregate default rate, monthly, 2010-01 .. 2024-12; last 24 months held out
// ─────────────────────────────────────────────────────────────────────────────
const int Months = 180;
const int Holdout = 24;
const int Train = Months - Holdout;
var start = new DateTime(2010, 1, 1);
var dates = Enumerable.Range(0, Months).Select(t => start.AddMonths(t)).ToArray();

// Unemployment: mean-reverting around 5.5% with a sharp 2020 shock that decays.
var unemployment = new double[Months];
var level = 5.5;
for (var t = 0; t < Months; t++)
{
    level = 5.5 + (0.92 * (level - 5.5)) + (0.12 * Gaussian());
    var shock = t >= 122 && t < 134 ? 4.5 * Math.Exp(-(t - 122) / 3.0) : 0d;
    unemployment[t] = level + shock;
}

// Default rate on the probit scale: z_t = a + b (U_t - 5.5) + ARMA(1,1) noise; DR = Phi(z).
var z = new double[Months];
var noise = 0d;
var lastInnovation = 0d;
for (var t = 0; t < Months; t++)
{
    var innovation = 0.045 * Gaussian();
    noise = (0.6 * noise) + innovation + (0.3 * lastInnovation);
    lastInnovation = innovation;
    z[t] = -2.05 + (0.13 * (unemployment[t] - 5.5)) + noise;
}

var rate = z.Select(NormalDistribution.Cdf).ToArray();
var trainZ = z.AsSpan(0, Train).ToArray();
var trainU = unemployment.AsSpan(0, Train).ToArray();
var futureU = unemployment.AsSpan(Train, Holdout).ToArray();

var stopwatch = Stopwatch.StartNew();
var auto = new AutoArima(new AutoArimaOptions
{
    MaxP = 3,
    MaxQ = 2,
    MaxPilotOrder = 12,
    Frequency = SeriesFrequency.Monthly,
});
var selection = auto.Select(trainZ, ExogenousMatrix.FromColumn(trainU));
var fit = selection.Best;
var selectionMs = stopwatch.Elapsed.TotalMilliseconds;

// Twenty-four months ahead, given the unemployment path that actually occurred.
var forecast = fit.Forecast(ForecastHorizon.Years(2), ExogenousMatrix.FromColumn(futureU));
var forecastRate = forecast.Mean.ToArray().Select(NormalDistribution.Cdf).ToArray();
var lowerRate = forecast.Lower.ToArray().Select(NormalDistribution.Cdf).ToArray();
var upperRate = forecast.Upper.ToArray().Select(NormalDistribution.Cdf).ToArray();
var actualRate = rate.AsSpan(Train, Holdout).ToArray();

var maeBps = 10_000 * ErrorMetrics.Mae(actualRate, forecastRate);
var coverage = ErrorMetrics.Coverage(actualRate, lowerRate, upperRate);
var kpss0 = selection.Differencing.First(c => c.Order == 0).Kpss;

Console.WriteLine("── 1. Aggregate default rate (probit scale, unemployment as regressor) ──");
Console.WriteLine($"   training months: {Train}, held out: {Holdout}, grid: {selection.Candidates.Count} models on one pass in {selectionMs:F0} ms");
Console.WriteLine($"   KPSS at d=0: {kpss0.Statistic:F3} (5% critical 0.463) → chose d={selection.ChosenDifferencing.Order}");
Console.WriteLine($"   selected: ARIMA{fit.Order}  AICc={fit.Diagnostics.Aicc:F1}  pilot m={fit.Diagnostics.PilotOrder}");
Console.WriteLine($"   phi=[{string.Join(", ", fit.AutoRegressive.ToArray().Select(v => v.ToString("F3", inv)))}]  theta=[{string.Join(", ", fit.MovingAverage.ToArray().Select(v => v.ToString("F3", inv)))}]  beta(unemp)={fit.ExogenousCoefficients.Span[0]:F3} (true 0.13)");
Console.WriteLine($"   Ljung-Box Q({fit.Diagnostics.LjungBoxLags})={fit.Diagnostics.LjungBoxStatistic:F2}  p={fit.Diagnostics.LjungBoxPValue:F3}  → residuals {(fit.Diagnostics.RejectsWhiteNoiseResiduals() ? "still autocorrelated" : "look like white noise")}");
Console.WriteLine($"   stationary: {fit.Diagnostics.IsStationary} (margin {fit.Diagnostics.StationarityMargin:F2})   invertible: {fit.Diagnostics.IsInvertible} (margin {fit.Diagnostics.InvertibilityMargin:F2})");
foreach (var warning in forecast.Warnings)
{
    Console.WriteLine($"   WARNING: {warning}");
}
Console.WriteLine($"   holdout: MAE {maeBps:F1} bps, 95% interval coverage {coverage:P0}");
Console.WriteLine();
Console.WriteLine("   ranked candidates (top 6):");
foreach (var c in selection.Candidates.Take(6))
{
    Console.WriteLine($"     ARIMA{c.Order}  AICc={c.Aicc,8:F1}  LB p={(c.Fit is null ? "  -  " : c.Fit.Diagnostics.LjungBoxPValue.ToString("F3", inv))}  {(c.IsAdmissible ? "admissible" : "NOT admissible")}");
}

Console.WriteLine();
Console.WriteLine("   month      actual   forecast    95% low   95% high");
for (var h = 0; h < Holdout; h += 3)
{
    Console.WriteLine($"   {dates[Train + h]:yyyy-MM}   {actualRate[h],7:P2}   {forecastRate[h],7:P2}   {lowerRate[h],7:P2}   {upperRate[h],7:P2}");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. A population of 300 accounts through the SQLite adapter, one grouped scan
// ─────────────────────────────────────────────────────────────────────────────
const int Accounts = 300;
var connectionString = "Data Source=file:demo?mode=memory&cache=shared";
using var keepAlive = new SqliteConnection(connectionString);
keepAlive.Open();
var truePhi = new Dictionary<string, double>(StringComparer.Ordinal);
long rowsInserted = 0;

using (var create = keepAlive.CreateCommand())
{
    create.CommandText = "CREATE TABLE balances (key TEXT, ts TEXT, bal REAL, log_bal REAL); CREATE INDEX ix ON balances(key, ts)";
    create.ExecuteNonQuery();
}

using (var tx = keepAlive.BeginTransaction())
using (var insert = keepAlive.CreateCommand())
{
    insert.Transaction = tx;
    insert.CommandText = "INSERT INTO balances VALUES (@k, @ts, @b, @lb)";
    var k = insert.Parameters.Add("@k", SqliteType.Text);
    var ts = insert.Parameters.Add("@ts", SqliteType.Text);
    var b = insert.Parameters.Add("@b", SqliteType.Real);
    var lb = insert.Parameters.Add("@lb", SqliteType.Real);

    for (var a = 0; a < Accounts; a++)
    {
        var key = $"ACC-{a + 1:D4}";
        var phi = 0.2 + (0.6 * rng.NextDouble());          // month-to-month persistence of balance changes
        truePhi[key] = phi;
        var length = 96 + rng.Next(0, 49);                    // 8 to 12 years of history
        var logBalance = Math.Log(5_000 + (50_000 * rng.NextDouble()));
        var change = 0d;
        var opened = start.AddMonths(rng.Next(0, Months - length));

        for (var t = 0; t < length; t++)
        {
            change = (phi * change) + (0.03 * Gaussian()) + 0.002;
            logBalance += change;
            k.Value = key;
            ts.Value = opened.AddMonths(t).ToString("O", inv);
            b.Value = Math.Exp(logBalance);
            lb.Value = logBalance;
            insert.ExecuteNonQuery();
            rowsInserted++;
        }
    }

    tx.Commit();
}

var population = new DbTimeSeriesSource(
    () => new SqliteConnection(connectionString),
    new DbSeriesQuery
    {
        CommandText = "SELECT key, ts, log_bal FROM balances ORDER BY key, ts",
        ValueColumn = "log_bal",
        KeyColumn = "key",
        TimeColumn = "ts",
        BatchSize = 1024,
    },
    // Monthly rows are 28–31 days apart: the step is nominal, the gap threshold is what matters.
    new DbSourceOptions { ExpectedStep = TimeSpan.FromDays(30), MaxGap = TimeSpan.FromDays(32) });

// Balances are modelled on the log scale, so a first difference is a monthly growth rate.
var accountModel = new ArimaModel(new ArimaOptions { Order = new(1, 1, 0), Frequency = SeriesFrequency.Monthly });

stopwatch.Restart();
var sequential = new List<KeyedArimaFit>();
await foreach (var r in accountModel.FitManyAsync(population, FitManyOptions.Sequential))
{
    sequential.Add(r);
}

var sequentialMs = stopwatch.Elapsed.TotalMilliseconds;

stopwatch.Restart();
var parallel = new List<KeyedArimaFit>();
await foreach (var r in accountModel.FitManyAsync(population, new FitManyOptions { DegreeOfParallelism = Environment.ProcessorCount }))
{
    parallel.Add(r);
}

var parallelMs = stopwatch.Elapsed.TotalMilliseconds;

var byKey = parallel.ToDictionary(r => r.Key, StringComparer.Ordinal);
var identical = sequential.All(s => byKey[s.Key].Fit is { } pf && s.Fit is { } sf
    && pf.AutoRegressive.Span[0] == sf.AutoRegressive.Span[0] && pf.Intercept == sf.Intercept);
var phis = parallel.Where(r => r.Succeeded).Select(r => r.Fit!.AutoRegressive.Span[0]).OrderBy(v => v).ToArray();
var lbRejections = parallel.Count(r => r.Succeeded && r.Fit!.Diagnostics.RejectsWhiteNoiseResiduals());
var phiError = parallel.Where(r => r.Succeeded).Average(r => Math.Abs(r.Fit!.AutoRegressive.Span[0] - truePhi[r.Key]));

Console.WriteLine();
Console.WriteLine("── 2. Population: 300 accounts from SQLite, one grouped scan ──");
Console.WriteLine($"   rows: {rowsInserted:N0}   fits: {parallel.Count(r => r.Succeeded)}/{parallel.Count} succeeded");
Console.WriteLine($"   sequential: {sequentialMs:F0} ms   parallel ({Environment.ProcessorCount} workers): {parallelMs:F0} ms   identical fits: {identical}");
Console.WriteLine($"   phi-hat across accounts: min {phis[0]:F2}  median {phis[phis.Length / 2]:F2}  max {phis[^1]:F2}   mean |phi-hat − phi| = {phiError:F3}");
Console.WriteLine($"   Ljung-Box rejections at 5%: {lbRejections}/{parallel.Count} (a correctly specified model expects about 5%)");

// ─────────────────────────────────────────────────────────────────────────────
// 3. Incremental: save after the training window, fold the held-out months later
// ─────────────────────────────────────────────────────────────────────────────
var incremental = new IncrementalArima(fit.Options, regressorCount: 1);
incremental.Fold(trainZ, ExogenousMatrix.FromColumn(trainU));
using var state = new MemoryStream();
incremental.SaveTo(state);
state.Position = 0;

var resumed = IncrementalArima.Restore(state, fit.Options);
resumed.Fold(z.AsSpan(Train, Holdout), ExogenousMatrix.FromColumn(futureU));
var folded = resumed.Solve();
var refit = new ArimaModel(fit.Options).Fit(z, ExogenousMatrix.FromColumn(unemployment));

Console.WriteLine();
Console.WriteLine("── 3. Incremental fold ──");
Console.WriteLine($"   saved state: {state.Length:N0} bytes after {Train} months; folded {Holdout} more");
Console.WriteLine($"   fold == full refit over 180 months: {folded.AutoRegressive.ToArray().SequenceEqual(refit.AutoRegressive.ToArray()) && folded.Intercept == refit.Intercept}");

// ─────────────────────────────────────────────────────────────────────────────
// Report
// ─────────────────────────────────────────────────────────────────────────────
File.WriteAllText(outPath, Report.Build(new Report.Model(
    dates, rate, Train, forecastRate, lowerRate, upperRate, fit, selection, kpss0, maeBps, coverage, selectionMs,
    Accounts, rowsInserted, parallel.Count(r => r.Succeeded), sequentialMs, parallelMs, Environment.ProcessorCount,
    phis, lbRejections, phiError, state.Length)));
Console.WriteLine();
Console.WriteLine($"report: {Path.GetFullPath(outPath)}");

/// <summary>The HTML report: one chart with crosshair and tooltip, its table view, and the numbers.</summary>
static class Report
{
    public sealed record Model(
        DateTime[] Dates, double[] Rate, int Train, double[] Forecast, double[] Lower, double[] Upper,
        ArimaFit Fit, ArimaSelection Selection, TimeSeries.Accumulators.KpssResult Kpss0, double MaeBps, double Coverage, double SelectionMs,
        int Accounts, long Rows, int Succeeded, double SequentialMs, double ParallelMs, int Workers,
        double[] Phis, int LbRejections, double PhiError, long StateBytes);

    public static string Build(Model m)
    {
        var inv = CultureInfo.InvariantCulture;
        var n = m.Dates.Length;
        var holdout = n - m.Train;

        // ---- geometry
        const double W = 960, H = 400, Left = 64, Right = 72, Top = 24, Bottom = 44;
        var plotW = W - Left - Right;
        var plotH = H - Top - Bottom;
        var all = m.Rate.Concat(m.Lower).Concat(m.Upper).ToArray();
        var yMax = Math.Ceiling(all.Max() * 100 * 2) / 2 / 100;   // to the next 0.5%
        var yMin = Math.Max(0, Math.Floor(all.Min() * 100 * 2) / 2 / 100);
        double X(int t) => Left + (plotW * t / (n - 1));
        double Y(double v) => Top + plotH - (plotH * (v - yMin) / (yMax - yMin));
        string F(double v) => v.ToString("0.##", inv);
        string Pct(double v) => (v * 100).ToString("0.00", inv) + "%";

        var observed = new StringBuilder();
        for (var t = 0; t <= m.Train; t++)
        {
            observed.Append(t == 0 ? "M" : "L").Append(F(X(t))).Append(' ').Append(F(Y(m.Rate[t]))).Append(' ');
        }

        var forecastPath = new StringBuilder();
        var band = new StringBuilder();
        for (var h = 0; h < holdout; h++)
        {
            var t = m.Train + h;
            forecastPath.Append(h == 0 ? $"M{F(X(m.Train))} {F(Y(m.Rate[m.Train]))} L" : "L").Append(F(X(t))).Append(' ').Append(F(Y(m.Forecast[h]))).Append(' ');
            band.Append(h == 0 ? "M" : "L").Append(F(X(t))).Append(' ').Append(F(Y(m.Upper[h]))).Append(' ');
        }

        for (var h = holdout - 1; h >= 0; h--)
        {
            band.Append('L').Append(F(X(m.Train + h))).Append(' ').Append(F(Y(m.Lower[h]))).Append(' ');
        }

        band.Append('Z');

        var actualDots = new StringBuilder();
        for (var h = 0; h < holdout; h++)
        {
            actualDots.Append($"<circle cx=\"{F(X(m.Train + h))}\" cy=\"{F(Y(m.Rate[m.Train + h]))}\" r=\"4\" class=\"dot\"/>");
        }

        var grid = new StringBuilder();
        var steps = (int)Math.Round((yMax - yMin) / 0.005);
        var every = Math.Max(1, steps / 5);
        for (var i = 0; i <= steps; i += every)
        {
            var v = yMin + (i * 0.005);
            grid.Append($"<line x1=\"{F(Left)}\" x2=\"{F(W - Right)}\" y1=\"{F(Y(v))}\" y2=\"{F(Y(v))}\" class=\"grid\"/>");
            grid.Append($"<text x=\"{F(Left - 8)}\" y=\"{F(Y(v) + 4)}\" class=\"tick\" text-anchor=\"end\">{(v * 100).ToString("0.0", inv)}%</text>");
        }

        var xTicks = new StringBuilder();
        for (var t = 0; t < n; t += 24)
        {
            xTicks.Append($"<text x=\"{F(X(t))}\" y=\"{F(H - 16)}\" class=\"tick\" text-anchor=\"middle\">{m.Dates[t].Year}</text>");
        }

        var originX = F(X(m.Train));
        var endLabelY = F(Y(m.Forecast[^1]));
        var series = JsonSerializer.Serialize(new
        {
            dates = m.Dates.Select(d => d.ToString("yyyy-MM", inv)),
            observed = m.Rate.Select((v, t) => t <= m.Train ? v : (double?)null),
            actual = m.Rate.Select((v, t) => t > m.Train ? v : (double?)null),
            forecast = Enumerable.Range(0, n).Select(t => t >= m.Train ? m.Forecast[t - m.Train] : (double?)null),
            lower = Enumerable.Range(0, n).Select(t => t >= m.Train ? m.Lower[t - m.Train] : (double?)null),
            upper = Enumerable.Range(0, n).Select(t => t >= m.Train ? m.Upper[t - m.Train] : (double?)null),
            left = Left, plotW, train = m.Train, n,
        });

        var candidates = new StringBuilder();
        foreach (var c in m.Selection.Candidates.Take(8))
        {
            candidates.Append($"<tr><td>ARIMA{c.Order}</td><td class=\"num\">{(c.Succeeded ? c.Aicc.ToString("F1", inv) : "—")}</td><td class=\"num\">{(c.Fit is null ? "—" : c.Fit.Diagnostics.LjungBoxPValue.ToString("F3", inv))}</td><td>{(c.IsAdmissible ? "yes" : "no")}</td></tr>");
        }

        var rows = new StringBuilder();
        for (var h = 0; h < holdout; h++)
        {
            var inside = m.Rate[m.Train + h] >= m.Lower[h] && m.Rate[m.Train + h] <= m.Upper[h];
            rows.Append($"<tr><td>{m.Dates[m.Train + h]:yyyy-MM}</td><td class=\"num\">{Pct(m.Rate[m.Train + h])}</td><td class=\"num\">{Pct(m.Forecast[h])}</td><td class=\"num\">{Pct(m.Lower[h])}</td><td class=\"num\">{Pct(m.Upper[h])}</td><td>{(inside ? "inside" : "outside")}</td></tr>");
        }

        var d = m.Fit.Diagnostics;
        var phi = string.Join(", ", m.Fit.AutoRegressive.ToArray().Select(v => v.ToString("F3", inv)));
        var theta = string.Join(", ", m.Fit.MovingAverage.ToArray().Select(v => v.ToString("F3", inv)));

        return $$"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TimeSeries.NET demo — default-rate forecast</title>
<style>
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb; --plane: #f9f9f7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6; --series-2: #eb6834;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1: #1a1a19; --plane: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
      --series-1: #3987e5; --series-2: #d95926;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1: #1a1a19; --plane: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926;
  }
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
  .key.s2 { background: var(--series-2); }
  .key.dot { width: 8px; height: 8px; border-radius: 50%; }
  .key.band { height: 10px; background: var(--series-2); opacity: .18; }
  svg { width: 100%; height: auto; display: block; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--axis); stroke-width: 1; }
  .tick, .lbl { fill: var(--muted); font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; font-variant-numeric: tabular-nums; }
  .lbl { fill: var(--text-secondary); }
  .observed { fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .forecast { fill: none; stroke: var(--series-2); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .band { fill: var(--series-2); opacity: .12; }
  .dot { fill: var(--series-1); stroke: var(--surface-1); stroke-width: 2; }
  .end { fill: var(--series-2); stroke: var(--surface-1); stroke-width: 2; }
  .crosshair { stroke: var(--axis); stroke-width: 1; display: none; }
  .tooltip { position: absolute; pointer-events: none; display: none; background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.08); min-width: 180px; }
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
  code { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--text-secondary); }
</style>
</head>
<body>
<main>
  <h1>Portfolio default rate — 24-month forecast</h1>
  <p class="sub">Monthly default rate modelled on the probit scale with unemployment as a regressor. Fitted on {{m.Train}} months to {{m.Dates[m.Train - 1]:MMM yyyy}}; the last {{holdout}} months are held out and compared. Order chosen by <code>AutoArima</code> on one pass; residuals checked by Ljung–Box.</p>

  <div class="tiles">
    <div class="tile"><div class="label">Selected model</div><div class="value">ARIMA{{m.Fit.Order}}</div><div class="note">AICc {{d.Aicc.ToString("F1", inv)}} · pilot m = {{d.PilotOrder}} · {{m.Selection.Candidates.Count}} candidates in {{m.SelectionMs.ToString("F0", inv)}} ms</div></div>
    <div class="tile"><div class="label">Unemployment effect (β)</div><div class="value">{{m.Fit.ExogenousCoefficients.Span[0].ToString("F3", inv)}}</div><div class="note">probit units per point · generated with 0.130</div></div>
    <div class="tile"><div class="label">Ljung–Box p-value</div><div class="value">{{d.LjungBoxPValue.ToString("F3", inv)}}</div><div class="note">Q({{d.LjungBoxLags}}) = {{d.LjungBoxStatistic.ToString("F2", inv)}}, df {{d.LjungBoxDegreesOfFreedom}} · {{(d.RejectsWhiteNoiseResiduals() ? "residuals still autocorrelated" : "residuals look like white noise")}}</div></div>
    <div class="tile"><div class="label">Held-out accuracy</div><div class="value">{{m.MaeBps.ToString("F1", inv)}} bps</div><div class="note">mean absolute error · 95% interval covered {{m.Coverage.ToString("P0", inv)}} of months</div></div>
  </div>

  <div class="card" id="chart-card">
    <div class="legend">
      <span><i class="key"></i>Observed</span>
      <span><i class="key dot"></i>Held-out actual</span>
      <span><i class="key s2"></i>Forecast</span>
      <span><i class="key band"></i>95% interval</span>
    </div>
    <svg viewBox="0 0 {{F(W)}} {{F(H)}}" role="img" aria-label="Default rate: observed history, held-out actuals, and the 24-month forecast with its 95% interval">
      {{grid}}
      <line x1="{{F(Left)}}" x2="{{F(W - Right)}}" y1="{{F(Top + plotH)}}" y2="{{F(Top + plotH)}}" class="axis"/>
      <line x1="{{originX}}" x2="{{originX}}" y1="{{F(Top)}}" y2="{{F(Top + plotH)}}" class="axis"/>
      <text x="{{F(X(m.Train) + 6)}}" y="{{F(Top + 12)}}" class="lbl">forecast origin</text>
      <path d="{{band}}" class="band"/>
      <path d="{{observed}}" class="observed"/>
      <path d="{{forecastPath}}" class="forecast"/>
      {{actualDots}}
      <circle cx="{{F(X(n - 1))}}" cy="{{endLabelY}}" r="4" class="end"/>
      <text x="{{F(X(n - 1) + 10)}}" y="{{F(Y(m.Forecast[^1]) + 4)}}" class="lbl">{{Pct(m.Forecast[^1])}}</text>
      {{xTicks}}
      <line id="crosshair" class="crosshair" y1="{{F(Top)}}" y2="{{F(Top + plotH)}}"/>
      <rect id="hit" x="{{F(Left)}}" y="{{F(Top)}}" width="{{F(plotW)}}" height="{{F(plotH)}}" fill="transparent"/>
    </svg>
    <div class="tooltip" id="tooltip"></div>
  </div>

  <div class="cols">
    <div>
      <h2>Held-out months — table view</h2>
      <table>
        <thead><tr><th>Month</th><th class="num">Actual</th><th class="num">Forecast</th><th class="num">95% low</th><th class="num">95% high</th><th>Interval</th></tr></thead>
        <tbody>{{rows}}</tbody>
      </table>
    </div>
    <div>
      <h2>Order selection</h2>
      <p class="sub">KPSS at d = 0: {{m.Kpss0.Statistic.ToString("F3", inv)}} against a 5% critical value of 0.463 → d = {{m.Selection.ChosenDifferencing.Order}}. Then (p, q) ranked by AICc on the same rows; a candidate that is not both stationary and invertible cannot win.</p>
      <table>
        <thead><tr><th>Candidate</th><th class="num">AICc</th><th class="num">Ljung–Box p</th><th>Admissible</th></tr></thead>
        <tbody>{{candidates}}</tbody>
      </table>
      <h2>Fitted coefficients</h2>
      <p class="sub"><code>φ = [{{phi}}]</code><br><code>θ = [{{theta}}]</code><br><code>c = {{m.Fit.Intercept.ToString("F4", inv)}}, σ̂² = {{m.Fit.InnovationVariance.ToString("E2", inv)}}</code></p>
    </div>
  </div>

  <h2>Population: {{m.Accounts}} accounts from SQLite in one grouped scan</h2>
  <div class="tiles">
    <div class="tile"><div class="label">Rows scanned</div><div class="value">{{m.Rows.ToString("N0", inv)}}</div><div class="note">{{m.Succeeded}} of {{m.Accounts}} accounts fitted · ARIMA(1,1,0) on log balances</div></div>
    <div class="tile"><div class="label">Sequential</div><div class="value">{{m.SequentialMs.ToString("F0", inv)}} ms</div><div class="note">one worker, results in key order</div></div>
    <div class="tile"><div class="label">Parallel</div><div class="value">{{m.ParallelMs.ToString("F0", inv)}} ms</div><div class="note">{{m.Workers}} workers, bit-identical fits</div></div>
    <div class="tile"><div class="label">Persistence φ̂</div><div class="value">{{m.Phis[m.Phis.Length / 2].ToString("F2", inv)}}</div><div class="note">median · range {{m.Phis[0].ToString("F2", inv)}} to {{m.Phis[^1].ToString("F2", inv)}} · mean |φ̂ − φ| {{m.PhiError.ToString("F3", inv)}}</div></div>
    <div class="tile"><div class="label">Ljung–Box rejections</div><div class="value">{{m.LbRejections}}</div><div class="note">of {{m.Accounts}} at 5% · about 5% expected when the model is right</div></div>
    <div class="tile"><div class="label">Incremental state</div><div class="value">{{m.StateBytes.ToString("N0", inv)}} B</div><div class="note">saved, restored, {{holdout}} months folded — equal to a full refit</div></div>
  </div>
</main>
<script type="application/json" id="series">{{series}}</script>
<script>
(function () {
  var data = JSON.parse(document.getElementById('series').textContent);
  var svg = document.querySelector('#chart-card svg');
  var hit = document.getElementById('hit');
  var cross = document.getElementById('crosshair');
  var tip = document.getElementById('tooltip');
  var card = document.getElementById('chart-card');
  function pct(v) { return v == null ? null : (v * 100).toFixed(2) + '%'; }
  function add(node) { if (node) tip.appendChild(node); }
  function row(label, value, cls) {
    if (value == null) return null;
    var r = document.createElement('div'); r.className = 'row';
    var l = document.createElement('span'); l.className = 'l';
    var k = document.createElement('i'); k.className = 'key ' + cls; l.appendChild(k); l.appendChild(document.createTextNode(label));
    var v = document.createElement('span'); v.className = 'v'; v.textContent = value;
    r.appendChild(l); r.appendChild(v); return r;
  }
  function show(evt) {
    var rect = svg.getBoundingClientRect();
    var scale = rect.width / svg.viewBox.baseVal.width;
    var x = (evt.clientX - rect.left) / scale;
    var t = Math.round((x - data.left) / data.plotW * (data.n - 1));
    t = Math.max(0, Math.min(data.n - 1, t));
    var cx = data.left + data.plotW * t / (data.n - 1);
    cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.style.display = 'block';
    while (tip.firstChild) tip.removeChild(tip.firstChild);
    var head = document.createElement('div'); head.className = 't'; head.textContent = data.dates[t]; tip.appendChild(head);
    add(row('Observed', pct(data.observed[t]), ''));
    add(row('Actual (held out)', pct(data.actual[t]), 'dot'));
    add(row('Forecast', pct(data.forecast[t]), 's2'));
    if (data.lower[t] != null) add(row('95% interval', pct(data.lower[t]) + ' – ' + pct(data.upper[t]), 'band'));
    tip.style.display = 'block';
    var px = cx * scale + (rect.left - card.getBoundingClientRect().left);
    var flip = px > rect.width * 0.7;
    tip.style.left = (flip ? px - tip.offsetWidth - 12 : px + 12) + 'px';
    tip.style.top = (rect.top - card.getBoundingClientRect().top + 40) + 'px';
  }
  hit.addEventListener('pointermove', show);
  hit.addEventListener('pointerleave', function () { cross.style.display = 'none'; tip.style.display = 'none'; });
})();
</script>
</body>
</html>
""";
    }
}
