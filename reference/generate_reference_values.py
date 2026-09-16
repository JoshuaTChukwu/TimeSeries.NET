#!/usr/bin/env python3
"""Reference forecasts from statsmodels, committed alongside their output.

Each case fixes a model's parameters, runs statsmodels' SARIMAX Kalman filter over a
seeded synthetic series with those parameters, and records the h-step forecast means and
standard errors. The C# tests rebuild the same fit from the same parameters and must
reproduce the numbers. This isolates the forecasting arithmetic — recursion, integration,
psi-weights — from estimator differences, since Hannan-Rissanen and exact MLE will not
agree on parameters but must agree on what a given model forecasts.

Model form (both libraries): on the differenced scale,
    z_t = c + phi_1 z_(t-1) + ... + e_t + theta_1 e_(t-1) + ... + beta' x_t
with additive MA signs. Exogenous cases are restricted to p = 0, where statsmodels'
regression-with-ARIMA-errors form and the library's ARMAX form coincide.

Regenerate with:
    python3 reference/generate_reference_values.py > reference/forecast_reference.json
"""
import json
import sys

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

N = 400
STEPS = 24


def simulate(rng, phi, theta, c, d, D, s, sigma, n, beta=None, x=None):
    """Simulate on the differenced scale in the library's model form, then integrate."""
    p, q = len(phi), len(theta)
    burn = 500
    e = rng.normal(0.0, sigma, n + burn)
    z = np.zeros(n + burn)
    xr = None
    if beta is not None:
        xr = np.concatenate([np.zeros((burn, x.shape[1])), x])
    for t in range(n + burn):
        v = c + e[t]
        for k in range(1, p + 1):
            if t - k >= 0:
                v += phi[k - 1] * z[t - k]
        for j in range(1, q + 1):
            if t - j >= 0:
                v += theta[j - 1] * e[t - j]
        if beta is not None:
            v += float(np.dot(beta, xr[t]))
        z[t] = v
    z = z[burn:]
    # integrate: seasonal first (as the library differences seasonal-first, integration
    # order is irrelevant for the final series since the operators commute).
    y = z.copy()
    for _ in range(D):
        acc = np.zeros(len(y) + s)
        for t in range(len(y)):
            acc[t + s] = acc[t] + y[t]
        y = acc[s:]
    for _ in range(d):
        y = np.concatenate([[0.0], np.cumsum(y)])[1:] if False else np.cumsum(y)
    return y


def case(name, order, seasonal, trend, c, phi, theta, sigma2, seed, beta=None):
    rng = np.random.default_rng(seed)
    p, d, q = order
    D, s = seasonal
    x = None
    if beta is not None:
        x = rng.normal(0.0, 1.0, (N + STEPS, len(beta)))
    y = simulate(rng, phi, theta, c, d, D, s, np.sqrt(sigma2), N,
                 beta=beta, x=None if x is None else x[:N])

    # Level of a differenced series is arbitrary; give it a recognisable one.
    y = y + 100.0

    exog = None if x is None else x[:N]
    model = SARIMAX(
        y,
        exog=exog,
        order=(p, d, q),
        seasonal_order=(0, D, 0, s) if D else (0, 0, 0, 0),
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    params = []
    if trend == "c":
        params.append(c)
    if beta is not None:
        params.extend(beta)
    params.extend(phi)
    params.extend(theta)
    params.append(sigma2)
    res = model.filter(np.array(params, dtype=float))
    fc = res.get_forecast(STEPS, exog=None if x is None else x[N:N + STEPS])

    return {
        "name": name,
        "order": list(order),
        "seasonal": list(seasonal),
        "include_intercept": trend == "c",
        "intercept": c if trend == "c" else 0.0,
        "phi": list(phi),
        "theta": list(theta),
        "beta": [] if beta is None else list(beta),
        "sigma2": sigma2,
        "series": [float(v) for v in y],
        "exogenous": None if x is None else [[float(v) for v in row] for row in x[:N]],
        "future_exogenous": None if x is None else [[float(v) for v in row] for row in x[N:N + STEPS]],
        "steps": STEPS,
        "forecast_mean": [float(v) for v in fc.predicted_mean],
        "forecast_se": [float(v) for v in fc.se_mean],
        "statsmodels_params": params,
    }


cases = [
    case("ar1_c",        (1, 0, 0), (0, 0), "c", 0.5, [0.7], [], 1.0, seed=1),
    case("ma1_c",        (0, 0, 1), (0, 0), "c", 0.2, [], [0.5], 1.0, seed=2),
    case("arma11_c",     (1, 0, 1), (0, 0), "c", 0.3, [0.6], [0.3], 1.5, seed=3),
    case("arma22_c",     (2, 0, 2), (0, 0), "c", 0.1, [0.5, 0.2], [0.4, 0.1], 0.8, seed=4),
    case("arima010",     (0, 1, 0), (0, 0), "n", 0.0, [], [], 2.0, seed=5),
    case("arima110",     (1, 1, 0), (0, 0), "n", 0.0, [0.5], [], 1.0, seed=6),
    case("arima110_c",   (1, 1, 0), (0, 0), "c", 0.1, [0.5], [], 1.0, seed=7),
    case("arima011",     (0, 1, 1), (0, 0), "n", 0.0, [], [-0.4], 1.0, seed=8),
    case("arima111",     (1, 1, 1), (0, 0), "n", 0.0, [0.4], [0.3], 1.0, seed=9),
    case("arima210",     (2, 1, 0), (0, 0), "n", 0.0, [0.3, 0.2], [], 1.0, seed=10),
    case("sarima110_010_12", (1, 1, 0), (1, 12), "n", 0.0, [0.4], [], 1.0, seed=11),
    case("armax_ma1_c",  (0, 0, 1), (0, 0), "c", 0.2, [], [0.5], 1.0, seed=12, beta=[2.0]),
    case("arimax011",    (0, 1, 1), (0, 0), "n", 0.0, [], [0.3], 1.0, seed=13, beta=[1.5, -0.5]),
]

json.dump({"generator": "statsmodels SARIMAX.filter(params).get_forecast",
           "n": N, "steps": STEPS, "cases": cases}, sys.stdout, indent=1)
