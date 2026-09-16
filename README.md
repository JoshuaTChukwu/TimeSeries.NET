# TimeSeries.NET

**TimeSeries.NET** is an open-source C#/.NET library for **time-series modeling and forecasting**.  
Starting with **ARIMAX**, this library is designed to grow into a full suite of forecasting tools—built natively for the .NET ecosystem.

---

## 🚀 Why TimeSeries.NET?

- 📊 Native .NET support for time-series forecasting
- 🔢 Implements models like ARIMA/ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables)
- 🧱 Designed for extensibility: more models coming (ETS, Holt-Winters, etc.)
- 🛠️ No external dependencies on `net8.0` (no Python bridges)
- 🌊 Fits from memory *or* streams straight from your database — nothing is materialised
- 🤝 Contributors are allowed commercial use (see license details)

---

## 📐 Design documents

The design is settled in two documents, and they are the authority — this README is a
summary and will lag behind them.

- [`docs/SPECIFICATION.html`](./docs/SPECIFICATION.html) — public API, estimation engine,
  forecasting, data access, numerical contract, milestones.
- [`docs/ARCHITECTURE.html`](./docs/ARCHITECTURE.html) — layering, the four contracts,
  state and lifetime, concurrency, decision log.

---

## 📎 Dependencies

| Target | Package dependencies |
| --- | --- |
| `net8.0` | **None.** Asserted by a unit test over `GetReferencedAssemblies()`. |
| `netstandard2.0` | `System.Memory`, `System.Threading.Tasks.Extensions`, `Microsoft.Bcl.AsyncInterfaces` — `Span<T>`, `ValueTask` and `IAsyncEnumerable` for .NET Framework 4.6.1+. All Microsoft-owned. |

The core package will never take a third-party reference. Satellites:

| Package | Depends on |
| --- | --- |
| `TimeSeries.NET.Data` | Nothing beyond the BCL — `System.Data.Common` covers Npgsql, SqlClient, Sqlite, MySqlConnector, Oracle, ClickHouse and DuckDB without referencing any of them. |
| `TimeSeries.NET.Data.MongoDb` | `MongoDB.Driver`. The one third-party reference in the set, which is why it is its own package. |
| `TimeSeries.NET.Extensions.DependencyInjection` | `Microsoft.Extensions.DependencyInjection.Abstractions`. |

---

## 📦 Installation

> _NuGet package coming soon_ – follow the repo to get notified when it's published!

---

## 🧪 Example usage

A complete program — this exact file is compiled and run against the packed packages as
part of the release check. It needs `TimeSeries.NET`, `TimeSeries.NET.Data` and, for the
SQLite part, `Microsoft.Data.Sqlite`.

```csharp
using Microsoft.Data.Sqlite;
using TimeSeries;
using TimeSeries.Data;

// A synthetic series: a random walk with drift, the kind of thing d = 1 is for.
var random = new Random(42);
var series = new double[2_000];
for (var t = 1; t < series.Length; t++)
    series[t] = series[t - 1] + 0.1 + random.NextDouble() - 0.5;

// One class covers ARIMA and ARIMAX. The estimator is stateless; the fit is immutable.
var model = new ArimaModel(new ArimaOptions
{
    Order = new(1, 1, 1),
    Frequency = SeriesFrequency.Monthly,
});

ArimaFit fit = model.Fit(series);
Console.WriteLine($"phi={fit.AutoRegressive.Span[0]:F3} theta={fit.MovingAverage.Span[0]:F3} drift={fit.Intercept:F3}");

// Forecast two years ahead. Intervals are never optional.
ForecastResult next = fit.Forecast(ForecastHorizon.Years(2));
Console.WriteLine($"h=1: {next.Mean.Span[0]:F2} [{next.Lower.Span[0]:F2}, {next.Upper.Span[0]:F2}]");
Console.WriteLine($"h=24: {next.Mean.Span[23]:F2} [{next.Lower.Span[23]:F2}, {next.Upper.Span[23]:F2}]");

// Total over the next year, with a standard error that includes the covariance between steps.
ForecastAggregate year = next.Sum(1, 12);
Console.WriteLine($"sum(h=1..12) = {year.Mean:F1} ± {year.StandardError:F1}");

// Let the library choose the order: d by a stationarity test, (p, q) by AICc, one pass.
ArimaSelection selection = new AutoArima(new AutoArimaOptions { MaxP = 3, MaxQ = 2, MaxPilotOrder = 12 }).Select(series);
Console.WriteLine($"auto: ARIMA{selection.Best.Order} (KPSS at d=0: {selection.Differencing[0].Kpss.Statistic:F2})");

// The same fit, streamed from a database through DbDataReader. Nothing is materialised.
var connectionString = "Data Source=file:readme?mode=memory&cache=shared";
using var keepAlive = new SqliteConnection(connectionString);
keepAlive.Open();
using (var create = keepAlive.CreateCommand())
{
    create.CommandText = "CREATE TABLE prices (ts TEXT, close REAL)";
    create.ExecuteNonQuery();
    using var insert = keepAlive.CreateCommand();
    insert.CommandText = "INSERT INTO prices VALUES (@ts, @close)";
    var ts = insert.Parameters.Add("@ts", SqliteType.Text);
    var close = insert.Parameters.Add("@close", SqliteType.Real);
    var day = new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc);
    for (var t = 0; t < series.Length; t++)
    {
        ts.Value = day.AddDays(t).ToString("O");
        close.Value = series[t];
        insert.ExecuteNonQuery();
    }
}

var source = new DbTimeSeriesSource(
    () => new SqliteConnection(connectionString),
    new DbSeriesQuery
    {
        CommandText = "SELECT ts, close FROM prices WHERE ts >= @from ORDER BY ts",
        ValueColumn = "close",
        TimeColumn = "ts",
        Parameters = { ["@from"] = "2020-01-01" },
    },
    new DbSourceOptions { ExpectedStep = TimeSpan.FromDays(1) });   // a gap throws, by default

ArimaFit streamed = await model.FitAsync(source);
Console.WriteLine($"streamed == in-memory: {streamed.AutoRegressive.Span[0] == fit.AutoRegressive.Span[0]}");

// New observations land later: fold them into saved state. Exactly equal to a batch refit.
var incremental = new IncrementalArima(model.Options);
incremental.Fold(series.AsSpan(0, 1_500));
using var state = new MemoryStream();
incremental.SaveTo(state);

state.Position = 0;
var resumed = IncrementalArima.Restore(state, model.Options);
resumed.Fold(series.AsSpan(1_500));
Console.WriteLine($"incremental == batch: {resumed.Solve().AutoRegressive.Span[0] == fit.AutoRegressive.Span[0]}");
```

Populations of series — one model per account, product, or client — come from the same
scan ordered by key then time: set `KeyColumn` on the query and use
`model.FitManyAsync(source)`. MongoDB collections are read the same way through
`TimeSeries.NET.Data.MongoDb`; `TimeSeries.NET.Extensions.DependencyInjection` adds
`services.AddTimeSeries()`, `AddArimaModel(options)` and `AddAutoArima(options)`.

Prediction intervals are not optional. An ARIMA point forecast converges to its drift
line within a handful of steps, so at a two-year horizon the mean path is a straight line
and the interval width is the entire content of the forecast. Forecast errors at
different steps are correlated, so to aggregate over a window — total sales next quarter
— use `ForecastResult.Sum(start, count)`, which carries the covariance, rather than
adding per-step variances.

### Order selection

`AutoArima` runs the whole grid on one pass. Differencing is chosen by a KPSS
stationarity test — information criteria cannot compare models fitted to differently
differenced series — and then `(p, q)` are ranked by AICc (or AIC/BIC) at that fixed
`d`, where every candidate is fitted to the same rows. The result carries the full
ranked table and every stationarity test, so the choice is inspectable.

### Model form

On the differenced scale `z_t = (1−B)^d (1−B^s)^D y_t` the fitted model is

```
z_t = c + φ₁ z_{t−1} + … + φ_p z_{t−p}  +  e_t + θ₁ e_{t−1} + … + θ_q e_{t−q}  +  β′x_t
```

with additive MA signs (as in statsmodels) and the regressors differenced by the same
operator as the series. This is the linear ARMAX form, which is what makes a one-pass
solve possible; it differs from "regression with ARIMA errors" (R's `Arima(xreg=)`),
where the AR polynomial also acts on `β′x_t`. In this form `β` is the contemporaneous
effect and the long-run effect is `β / (1 − Σφ)`.

### Reading from views — the access model

The library reads; it never writes, never generates SQL, and never needs a table. The
database owner creates a view over whatever schema they have and grants `SELECT` on it:

```sql
CREATE VIEW vw_debt_history   AS SELECT period, debt_gdp, unemployment, inflation, growth, rate FROM fiscal_quarterly;
CREATE VIEW vw_macro_scenario AS SELECT scenario, period, unemployment, inflation, growth, rate FROM macro_projection;
GRANT SELECT ON vw_debt_history, vw_macro_scenario TO forecasting_role;
```

The forecasting process connects as `forecasting_role`. Its `DbSeriesQuery` names the
view and the roles of its columns; the connection is opened for one traversal, read
with sequential access, and disposed before any arithmetic starts.

An ARIMAX forecast needs the regressors' **future** values, and the library will not
extrapolate them. They come from a second view — a scenario table of assumed paths —
read through the same adapter with `ValueColumn = null` and
`ExogenousMatrix.FromSourceAsync(source)`. Several scenarios (baseline, adverse) are
several forecasts from one fit. `samples/TimeSeries.NET.PublicDebt` is this workflow
end to end: debt/GDP fitted from one view, ten years forecast under two scenarios from
another.

### Regressors on a differenced series

With `d = 1` the regressors are, by default, differenced with the series
(`RegressorDifferencing.SameAsSeries` — the regression-with-ARIMA-errors convention:
changes explained by changes). When the *level* of a driver acts on the *change* in
the series — public debt grows by the level of the deficit, rates and growth; default
rates move with the level of unemployment — set `RegressorDifferencing.None`, and the
model becomes Δy_t = c + φ Δy_{t−1} + β′x_t + ε_t with x undifferenced. Future
regressors are then supplied as levels too.

### Gaps and ordering

ARIMA assumes regular spacing. `DbTimeSeriesSource` checks it when a `TimeColumn` and an
`ExpectedStep` are given: a gap **throws** by default (`SeriesGapException`, naming the
timestamps either side); `ForwardFill` and `Interpolate` are opt-in. A timestamp that does
not advance, or a key that arrives out of order in a grouped scan, throws
`SeriesOrderException`. A silently misaligned fit is the one outcome the library refuses
to produce.

## 🧭 Roadmap

Milestones are defined in [the specification](./docs/SPECIFICATION.html) §13, each with
its own acceptance criteria.

- [x] **M1** — Project restructure, namespace rename, unified `DifferenceSpec(d, D, period)`
      streaming transform with `IntegrationState`
- [x] **M2** — Accumulator layer: lag-augmented Gram matrix, classical autocovariances,
      moments; `Add` / `Merge` / `Scale` / `Freeze`
- [x] **M3** — Solver and diagnostics: normal equations, Durbin–Levinson PACF, error
      metrics (MAE, MSE, RMSE, MAPE, sMAPE)
- [x] **M4** — ARIMA and ARIMAX fitting from the Gram matrix
- [x] **M5** — Forecasting: ψ-weights, integration, prediction intervals
- [x] **M6** — Streaming: batched cursor, `DbTimeSeriesSource`, multi-series scan,
      incremental fold
- [x] **M7** — `AutoArima` grid search, MongoDB source, DI extensions, NuGet packaging
- [x] Ljung–Box in one pass; parallel per-key workers; admissibility guard in `AutoArima`; `RegressorDifferencing`
- [x] Samples: `samples/TimeSeries.NET.Demo` (default rate, population scan) and `samples/TimeSeries.NET.PublicDebt` (two views, two scenarios)
- [ ] Publish `0.1.0-preview.1` to NuGet.org
- [ ] Additional models: ETS, Holt–Winters

### Why one pass

Hannan–Rissanen is usually described as two regressions in sequence. Both stages are
linear in the same lagged values, so a single lag-augmented Gram matrix serves both — one
traversal regardless of `p`, `q` and `m`. Because that matrix is a pure sum, an
incremental fold is *exactly* a batch refit, and an entire `(p,d,q)` grid search costs
nothing after the first scan.

---

## 🔗 Follow the Journey

- 📺 **Devlog series on YouTube**: [Joshua Chukwu – YouTube](https://www.youtube.com/@joshuatchukwu)  
- 💼 **Professional updates on LinkedIn**: [Connect with me on LinkedIn](https://www.linkedin.com/in/joshua-chukwu-653196192/)  
- ⭐️ **Star this repo to stay updated with progress!**

---

## 🤝 Contributing

Want to contribute to **TimeSeries.NET**?  
We welcome PRs, ideas, and issue reports.  
**Meaningful contributors will be granted a commercial-use license.**

Check out `CONTRIBUTING.md` (coming soon) or open an issue to get started.

---

## 📄 License

This project is licensed under a **Custom Non-Commercial License**:

- ✅ Free for personal, academic, and research use  
- 💼 Commercial use requires a paid license  
- 🧑‍💻 Contributors may use the library commercially (see `LICENSE.txt`)  

📬 Contact: jchukwu61@gmail.com  
📎 Read full license: [LICENSE.txt](./LICENSE.txt)