using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Estimation;

/// <summary>
/// The property Hannan-Rissanen actually promises: consistency. Estimates from n = 10^4
/// land within two standard errors of the truth, and the error shrinks as n grows.
/// Seeds are fixed, so a failure reproduces.
/// </summary>
public class ParameterRecoveryTests
{
    private const int N = 10_000;

    [Fact]
    public void Ar1_RecoversPhi_WithinTwoTheoreticalStandardErrors()
    {
        const double Phi = 0.7;
        var theoreticalSe = Math.Sqrt((1d - (Phi * Phi)) / N);
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) });

        var fit = model.Fit(SeriesGenerator.Ar1(N, Phi, seed: 1));
        Assert.Equal(Phi, fit.AutoRegressive.Span[0], 2d * theoreticalSe);

        // The reported standard error should agree with the asymptotic one.
        Assert.Equal(theoreticalSe, fit.StandardErrors.Span[1], theoreticalSe * 0.25);

        // Coverage over many seeds: a 2-SE band should catch about 95%.
        var inside = 0;
        for (var seed = 1; seed <= 20; seed++)
        {
            var estimate = model.Fit(SeriesGenerator.Ar1(N, Phi, seed)).AutoRegressive.Span[0];
            if (Math.Abs(estimate - Phi) < 2d * theoreticalSe)
            {
                inside++;
            }
        }

        Assert.True(inside >= 17, $"Only {inside}/20 AR(1) estimates within two standard errors.");
    }

    [Fact]
    public void Ma1_RecoversTheta_WithinTwoReportedStandardErrors()
    {
        const double Theta = 0.5;
        var model = new ArimaModel(new ArimaOptions { Order = new(0, 0, 1) });

        var inside = 0;
        var worst = 0d;

        for (var seed = 1; seed <= 20; seed++)
        {
            var fit = model.Fit(SeriesGenerator.Arma(N, [], [Theta], seed));
            var estimate = fit.MovingAverage.Span[0];
            var se = fit.StandardErrors.Span[1];

            Assert.True(fit.Diagnostics.IsInvertible);
            worst = Math.Max(worst, Math.Abs(estimate - Theta));

            if (Math.Abs(estimate - Theta) < 2d * se)
            {
                inside++;
            }
        }

        Assert.True(inside >= 16, $"Only {inside}/20 MA(1) estimates within two reported standard errors.");
        Assert.True(worst < 0.05, $"Worst MA(1) error over 20 seeds was {worst}.");
    }

    [Fact]
    public void Arma11_RecoversAllThreeCoefficients()
    {
        const double Phi = 0.6;
        const double Theta = 0.3;
        const double Constant = 1.0;
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1) });

        var fit = model.Fit(SeriesGenerator.Arma(N, [Phi], [Theta], seed: 7, constant: Constant));

        Assert.Equal(Phi, fit.AutoRegressive.Span[0], 2.5 * fit.StandardErrors.Span[1]);
        Assert.Equal(Theta, fit.MovingAverage.Span[0], 2.5 * fit.StandardErrors.Span[2]);
        Assert.Equal(Constant, fit.Intercept, 2.5 * fit.StandardErrors.Span[0]);
        Assert.Equal(1d, fit.InnovationVariance, 0.05);
        Assert.True(fit.Diagnostics.IsStationary);
        Assert.True(fit.Diagnostics.IsInvertible);
        Assert.True(fit.Diagnostics.PilotOrder >= 2);
    }

    [Fact]
    public void Armax_RecoversKnownBeta()
    {
        const double Phi = 0.5;
        const double Beta = 2.0;
        const double Constant = 1.0;
        var x = SeriesGenerator.Ar1(N, 0.4, seed: 55);
        var noise = SeriesGenerator.Arma(N, [Phi], [], seed: 56);
        var y = new double[N];

        // y_t = c + phi y_(t-1) + beta x_t + e_t, generated exactly in the fitted form.
        for (var t = 0; t < N; t++)
        {
            y[t] = Constant + (t > 0 ? Phi * y[t - 1] : 0d) + (Beta * x[t]) + (noise[t] - (t > 0 ? Phi * noise[t - 1] : 0d));
        }

        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).Fit(y, ExogenousMatrix.FromColumn(x));

        Assert.Equal(Beta, fit.ExogenousCoefficients.Span[0], 2d * fit.StandardErrors.Span[2]);
        Assert.Equal(Phi, fit.AutoRegressive.Span[0], 2.5 * fit.StandardErrors.Span[1]);
        Assert.Equal(Constant, fit.Intercept, 2.5 * fit.StandardErrors.Span[0]);
        Assert.Equal(1, fit.RegressorCount);
    }

    [Fact]
    public void Armax_WithTwoRegressorsAndAnMaTerm_RecoversBothBetas()
    {
        const double Beta1 = 1.5;
        const double Beta2 = -0.75;
        var x1 = SeriesGenerator.Ar1(N, 0.3, seed: 61);
        var x2 = SeriesGenerator.Gaussian(N, seed: 62);
        var noise = SeriesGenerator.Arma(N, [0.5], [0.4], seed: 63);
        var y = new double[N];

        for (var t = 0; t < N; t++)
        {
            y[t] = noise[t] + (Beta1 * x1[t]) + (Beta2 * x2[t]);
        }

        // Regressors enter additively on the noise here, so the ARMA part absorbs the
        // dynamics and the betas are identified directly.
        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1) })
            .Fit(y, ExogenousMatrix.FromColumns(x1, x2));

        Assert.Equal(2, fit.RegressorCount);
        Assert.Equal(Beta1, fit.ExogenousCoefficients.Span[0], 0.1);
        Assert.Equal(Beta2, fit.ExogenousCoefficients.Span[1], 0.1);
    }

    [Fact]
    public void Arima110_RecoversPhiAndDriftOnTheDifferencedScale()
    {
        const double Phi = 0.5;
        const double Drift = 0.1;
        var differenced = SeriesGenerator.Arma(N, [Phi], [], seed: 9, constant: Drift);
        var series = SeriesGenerator.Integrate(differenced, order: 1, level: 100d);

        var fit = new ArimaModel(new ArimaOptions { Order = new(1, 1, 0) }).Fit(series);

        Assert.Equal(Phi, fit.AutoRegressive.Span[0], 2.5 * fit.StandardErrors.Span[1]);
        Assert.Equal(Drift, fit.Intercept, 2.5 * fit.StandardErrors.Span[0]);
        Assert.Equal(N + 1, fit.ObservationCount);
        Assert.Equal(new FitWindow(0, N + 1), fit.Window);
    }

    [Fact]
    public void LevelRegressorsOnADifferencedSeries_RecoverBeta_WhateverTheBatching()
    {
        // dy_t = 0.1 + 0.4 dy_(t-1) + 0.8 u_t + e_t with u_t a stationary level: the debt
        // dynamics shape. Differencing u as well would put beta on du_t and miss.
        const double Beta = 0.8;
        var u = SeriesGenerator.Ar1(N + 1, 0.6, seed: 71);
        var e = SeriesGenerator.Gaussian(N + 1, seed: 72);
        var dy = new double[N + 1];
        for (var t = 1; t <= N; t++)
        {
            dy[t] = 0.1 + (0.4 * dy[t - 1]) + (Beta * u[t]) + e[t];
        }

        var y = SeriesGenerator.Integrate(dy.Skip(1).ToArray(), 1, level: 50);
        var x = ExogenousMatrix.FromColumn(u);
        var options = new ArimaOptions { Order = new(1, 1, 0), RegressorDifferencing = RegressorDifferencing.None };

        var fit = new ArimaModel(options).Fit(y, x);
        Assert.Equal(Beta, fit.ExogenousCoefficients.Span[0], 2.5 * fit.StandardErrors.Span[2]);
        Assert.Equal(0.4, fit.AutoRegressive.Span[0], 2.5 * fit.StandardErrors.Span[1]);

        // Alignment survives batch boundaries: the level regressor must be read at the raw
        // row its differenced value came from, whatever the batch size.
        var scan = new ArimaScan(options, regressorCount: 1);
        for (var start = 0; start < y.Length; start += 7)
        {
            var take = Math.Min(7, y.Length - start);
            scan.Accept(y.AsSpan(start, take), x.Rows(start, take));
        }

        var batched = scan.Solve(new FitWindow(0, y.Length));
        Assert.Equal(fit.ExogenousCoefficients.ToArray(), batched.ExogenousCoefficients.ToArray());

        // The other convention fits the same data without error — it answers a different
        // question (changes on changes), so its beta is not compared with this one.
        var differenced = new ArimaModel(options with { RegressorDifferencing = RegressorDifferencing.SameAsSeries }).Fit(y, x);
        Assert.True(differenced.Diagnostics.IsStationary);

        // Forecast: level regressors pass through undifferenced.
        var future = ExogenousMatrix.FromColumn([1.0, 1.0]);
        var forecast = fit.Forecast(ForecastHorizon.Periods(2), future);
        var expectedStep1 = y[^1] + fit.Intercept + (fit.AutoRegressive.Span[0] * (y[^1] - y[^2])) + (fit.ExogenousCoefficients.Span[0] * 1.0);
        Assert.Equal(expectedStep1, forecast.Mean.Span[0], 1e-9);
    }

    [Fact]
    public void ErrorShrinksAsTheSeriesGrows()
    {
        const double Phi = 0.6;
        const double Theta = 0.3;
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 1), PilotOrder = 8 });

        double MeanAbsoluteError(int n)
        {
            var total = 0d;
            for (var seed = 1; seed <= 8; seed++)
            {
                var fit = model.Fit(SeriesGenerator.Arma(n, [Phi], [Theta], seed));
                total += Math.Abs(fit.AutoRegressive.Span[0] - Phi) + Math.Abs(fit.MovingAverage.Span[0] - Theta);
            }

            return total / 8;
        }

        var small = MeanAbsoluteError(500);
        var large = MeanAbsoluteError(20_000);

        Assert.True(large < small / 2, $"Error at n=20000 ({large}) is not well below error at n=500 ({small}).");
    }
}
