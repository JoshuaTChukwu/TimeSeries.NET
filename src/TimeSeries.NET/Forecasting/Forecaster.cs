using TimeSeries.Forecasting;
using TimeSeries.Transforms;

namespace TimeSeries;

/// <summary>
/// The forecast recursion: point forecasts on the differenced scale, integration back to
/// the original scale, and ψ-weight intervals. Pure functions of an <see cref="ArimaFit"/>.
/// </summary>
internal static class Forecaster
{
    internal static ForecastResult Forecast(ArimaFit fit, int steps, ExogenousMatrix? future, ForecastOptions options)
    {
        options.Validate();

        var p = fit.Order.P;
        var q = fit.Order.Q;
        var r = fit.RegressorCount;
        var phi = fit.AutoRegressive.Span;
        var theta = fit.MovingAverage.Span;
        var beta = fit.ExogenousCoefficients.Span;
        var seed = fit.Seed;

        if (r > 0 && future is null)
        {
            throw new ArgumentException(
                $"The model has {r} exogenous regressor(s); their future values over the {steps}-step horizon " +
                "are required. The library does not extrapolate the caller's regressors.",
                nameof(future));
        }

        if (r == 0 && future is not null)
        {
            throw new ArgumentException("The model has no exogenous regressors; future regressor values do not apply.", nameof(future));
        }

        if (future is not null)
        {
            if (future.RegressorCount != r)
            {
                throw new ArgumentException(
                    $"The model was fitted with {r} regressor(s) but {future.RegressorCount} were supplied.",
                    nameof(future));
            }

            if (future.Count != steps)
            {
                throw new ArgumentException(
                    $"A {steps}-step forecast needs {steps} rows of future regressors; {future.Count} supplied.",
                    nameof(future));
            }
        }

        var warnings = new List<string>();

        if (!fit.Diagnostics.IsInvertible)
        {
            const string Message =
                "The moving-average polynomial is not invertible, so the innovations feeding the recursion " +
                "are not identified and the prediction intervals should not be trusted.";

            if (options.RequireInvertible)
            {
                throw new InvalidOperationException(Message);
            }

            warnings.Add(Message);
        }

        if (!fit.Diagnostics.IsStationary && fit.Order.P > 0)
        {
            warnings.Add(
                "The autoregressive polynomial is not stationary on the differenced scale; the forecast will " +
                "diverge rather than revert. Consider a higher differencing order.");
        }

        // Future regressors, differenced exactly as the fitted ones were, continuing from
        // the retained tail of each column.
        var futureDifferenced = DifferenceFuture(future, seed, steps, r);

        // Point forecasts on the differenced scale. The recursion reads observed values
        // and residuals while they exist, and its own forecasts (with zero future
        // innovations) after that.
        var recent = seed.RecentValues;
        var residuals = seed.RecentResiduals;
        var path = new double[steps];

        for (var h = 1; h <= steps; h++)
        {
            var value = fit.Intercept;

            for (var k = 1; k <= p; k++)
            {
                var index = h - k;
                value += phi[k - 1] * (index >= 1 ? path[index - 1] : recent[p + index - 1]);
            }

            for (var j = 1; j <= q; j++)
            {
                var index = h - j;

                if (index < 1)
                {
                    value += theta[j - 1] * residuals[q + index - 1];
                }
            }

            for (var i = 0; i < r; i++)
            {
                value += beta[i] * futureDifferenced[((h - 1) * r) + i];
            }

            path[h - 1] = value;
        }

        var mean = seed.Integration.Integrate(path);

        var psi = PsiWeights.Integrate(PsiWeights.Arma(phi, theta, steps), fit.Options.Differencing);
        var standardErrors = PsiWeights.StandardErrors(psi, fit.InnovationVariance);

        return new ForecastResult(
            mean, standardErrors, psi, fit.InnovationVariance, options.Confidence, fit.Window.To, warnings.ToArray());
    }

    private static double[] DifferenceFuture(ExogenousMatrix? future, ForecastSeed seed, int steps, int r)
    {
        var differenced = new double[steps * r];

        if (future is null || r == 0)
        {
            return differenced;
        }

        var column = new double[steps];
        var output = new double[steps];

        for (var i = 0; i < r; i++)
        {
            for (var t = 0; t < steps; t++)
            {
                column[t] = future[t, i];
            }

            SeriesGuard.Finite(column, nameof(future));

            var transform = DifferenceTransform.Continue(seed.RegressorStates[i]);
            var produced = transform.Transform(column, output);

            if (produced != steps)
            {
                throw new InvalidOperationException("Internal error: a primed transform did not produce one output per input.");
            }

            for (var t = 0; t < steps; t++)
            {
                differenced[(t * r) + i] = output[t];
            }
        }

        return differenced;
    }
}
