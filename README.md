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

The core package will never take a third-party reference. Database drivers live in
satellite packages (`TimeSeries.NET.Data` over `DbDataReader`, which covers Npgsql,
SqlClient, Sqlite, MySqlConnector, Oracle, ClickHouse and DuckDB without referencing any
of them).

---

## 📦 Installation

> _NuGet package coming soon_ – follow the repo to get notified when it's published!

---

## 🧪 Example usage

The API below is implemented on the `main` line of work (M1–M6); packaging (M7) is
what remains before a NuGet release.

One class covers both ARIMA and ARIMAX — they are the same model, and exogenous
regressors are simply optional. The estimator is stateless and the fit it returns is
immutable, holds no data, and serialises to a few hundred bytes.

```csharp
var model = new ArimaModel(new ArimaOptions
{
    Order = new(2, 1, 1),
    Frequency = SeriesFrequency.BusinessDaily,
});

// In memory, for ordinary data.
ArimaFit fit = model.Fit(closingPrices);

// Or streamed from your own database, never materialising the series.
// TimeSeries.NET.Data reaches any ADO.NET provider through DbDataReader.
var source = new DbTimeSeriesSource(
    () => new NpgsqlConnection(connectionString),
    new DbSeriesQuery
    {
        CommandText = "SELECT ts, close_px, vix FROM market.daily_bars WHERE symbol = @symbol ORDER BY ts",
        ValueColumn = "close_px",
        ExogenousColumns = ["vix"],
        TimeColumn = "ts",
        Parameters = { ["symbol"] = "EURUSD" },
    },
    new DbSourceOptions { ExpectedStep = TimeSpan.FromDays(1), MaxGap = TimeSpan.FromDays(4) });

ArimaFit streamed = await model.FitAsync(source, ct);

// A population of series in one ordered scan — set KeyColumn and ORDER BY key, ts.
await foreach (KeyedArimaFit perAccount in model.FitManyAsync(groupedSource, ct)) { /* ... */ }

// New observations land: fold them into saved state. Exactly equal to a batch refit.
var incremental = IncrementalArima.Restore(savedState, model.Options);
await incremental.FoldAsync(newRowsSource, ct);
ArimaFit refreshed = incremental.Solve();

ForecastResult next = fit.Forecast(ForecastHorizon.Years(2), futureRegressors);

Console.WriteLine($"{next.Mean.Span[0]:F4} [{next.Lower.Span[0]:F4}, {next.Upper.Span[0]:F4}]");
```

Prediction intervals are not optional. An ARIMA point forecast converges to its drift
line within a handful of steps, so at a two-year horizon the mean path is a straight line
and the interval width is the entire content of the forecast. Forecast errors at
different steps are correlated, so to aggregate over a window — total sales next quarter
— use `ForecastResult.Sum(start, count)`, which carries the covariance, rather than
adding per-step variances.

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
- [ ] **M7** — `AutoArima` grid search, MongoDB source, DI extensions, NuGet packaging
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