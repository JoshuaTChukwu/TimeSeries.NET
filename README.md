# TimeSeries.NET

**TimeSeries.NET** is an open-source C#/.NET library for **time-series modeling and forecasting**.  
Starting with **ARIMAX**, this library is designed to grow into a full suite of forecasting tools—built natively for the .NET ecosystem.

---

## 🚀 Why TimeSeries.NET?

- 📊 Native .NET support for time-series forecasting
- 🔢 Implements models like ARIMA/ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables)
- 🧱 Designed for extensibility: more models coming (ETS, Holt-Winters, etc.)
- 🛠️ No external dependencies (no Python bridges)
- 🤝 Contributors are allowed commercial use (see license details)

---

## 📦 Installation

> _NuGet package coming soon_ – follow the repo to get notified when it's published!

---

## 🧪 Example Usage (Coming Soon)

```csharp
var model = new ArimaxModel(order: (1, 1, 1));
model.Fit(y: timeSeriesData, exogenous: macroVariables);
var forecast = model.Forecast(steps: 3);
```

## 🧭 Roadmap

- [x] Project Initialization  
- [ ] Time Series Utilities (differencing, ACF, PACF)  
- [ ] ARIMA Core  
- [ ] ARIMAX: External Regressors  
- [ ] Forecasting Engine  
- [ ] Evaluation Metrics (MAE, RMSE, MAPE)  
- [ ] Additional Models: ETS, Holt-Winters, etc.  
- [ ] NuGet Packaging  
- [ ] Sample Use Cases & Docs  
- [ ] Web Dashboard / Visualizer (future)  

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