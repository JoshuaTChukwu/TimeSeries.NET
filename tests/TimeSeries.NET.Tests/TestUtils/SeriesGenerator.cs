namespace TimeSeries.Tests.TestUtils;

/// <summary>
/// Seeded series generation. Every generator takes an explicit seed so a failure is
/// reproducible rather than flaky — a simulated-series test that cannot be re-run with
/// the same data is not a test.
/// </summary>
public static class SeriesGenerator
{
    /// <summary>Uniform noise on [-1, 1), scaled.</summary>
    public static double[] Uniform(int length, int seed, double scale = 1d)
    {
        var random = new Random(seed);
        var series = new double[length];

        for (var i = 0; i < length; i++)
        {
            series[i] = ((random.NextDouble() * 2d) - 1d) * scale;
        }

        return series;
    }

    /// <summary>
    /// A random walk with drift, which is what d = 1 is meant to remove.
    /// </summary>
    public static double[] RandomWalk(int length, int seed, double drift = 0d, double level = 0d)
    {
        var random = new Random(seed);
        var series = new double[length];
        var value = level;

        for (var i = 0; i < length; i++)
        {
            value += drift + ((random.NextDouble() * 2d) - 1d);
            series[i] = value;
        }

        return series;
    }

    /// <summary>
    /// A stationary AR(1): <c>y_t = phi * y_{t-1} + e_t</c> with Gaussian innovations.
    /// Its theoretical autocorrelation is <c>rho_k = phi^k</c>, which is checkable
    /// without any reference implementation.
    /// </summary>
    /// <param name="length">Observations to return.</param>
    /// <param name="phi">The AR coefficient; must be inside the unit circle.</param>
    /// <param name="seed">PRNG seed.</param>
    /// <param name="sigma">Innovation standard deviation.</param>
    /// <param name="burnIn">
    /// Observations generated and discarded so the returned series starts from the
    /// stationary distribution rather than from zero.
    /// </param>
    public static double[] Ar1(int length, double phi, int seed, double sigma = 1d, int burnIn = 1000)
    {
        var normals = Gaussian(length + burnIn, seed, sigma);
        var value = 0d;

        for (var i = 0; i < burnIn; i++)
        {
            value = (phi * value) + normals[i];
        }

        var series = new double[length];

        for (var i = 0; i < length; i++)
        {
            value = (phi * value) + normals[burnIn + i];
            series[i] = value;
        }

        return series;
    }

    /// <summary>
    /// A stationary AR(p): <c>y_t = sum_k phi_k y_(t-k) + e_t</c> with Gaussian innovations.
    /// The caller is responsible for choosing coefficients inside the stationary region.
    /// </summary>
    public static double[] Ar(int length, double[] phi, int seed, double sigma = 1d, int burnIn = 1000)
    {
        var normals = Gaussian(length + burnIn, seed, sigma);
        var history = new double[phi.Length];
        var series = new double[length];

        for (var i = 0; i < length + burnIn; i++)
        {
            var value = normals[i];

            for (var k = 0; k < phi.Length; k++)
            {
                value += phi[k] * history[k];
            }

            for (var k = phi.Length - 1; k > 0; k--)
            {
                history[k] = history[k - 1];
            }

            if (phi.Length > 0)
            {
                history[0] = value;
            }

            if (i >= burnIn)
            {
                series[i - burnIn] = value;
            }
        }

        return series;
    }

    /// <summary>
    /// A stationary, invertible ARMA(p, q) with an optional constant, in the library's
    /// additive MA convention:
    /// <c>y_t = c + sum_k phi_k y_(t-k) + e_t + sum_j theta_j e_(t-j)</c>.
    /// </summary>
    public static double[] Arma(
        int length, double[] phi, double[] theta, int seed, double constant = 0d, double sigma = 1d, int burnIn = 1000)
    {
        var innovations = Gaussian(length + burnIn, seed, sigma);
        var yHistory = new double[phi.Length];
        var eHistory = new double[theta.Length];
        var series = new double[length];

        for (var i = 0; i < length + burnIn; i++)
        {
            var e = innovations[i];
            var value = constant + e;

            for (var k = 0; k < phi.Length; k++)
            {
                value += phi[k] * yHistory[k];
            }

            for (var j = 0; j < theta.Length; j++)
            {
                value += theta[j] * eHistory[j];
            }

            Shift(yHistory, value);
            Shift(eHistory, e);

            if (i >= burnIn)
            {
                series[i - burnIn] = value;
            }
        }

        return series;
    }

    /// <summary>Integrates a series <paramref name="order"/> times from the given starting level.</summary>
    public static double[] Integrate(double[] differenced, int order, double level = 0d)
    {
        var current = differenced;

        for (var d = 0; d < order; d++)
        {
            var integrated = new double[current.Length + 1];
            integrated[0] = level;

            for (var t = 1; t < integrated.Length; t++)
            {
                integrated[t] = integrated[t - 1] + current[t - 1];
            }

            current = integrated;
        }

        return current;
    }

    private static void Shift(double[] history, double newest)
    {
        for (var k = history.Length - 1; k > 0; k--)
        {
            history[k] = history[k - 1];
        }

        if (history.Length > 0)
        {
            history[0] = newest;
        }
    }

    /// <summary>Gaussian draws by Box-Muller from a seeded uniform source.</summary>
    public static double[] Gaussian(int length, int seed, double sigma = 1d)
    {
        var random = new Random(seed);
        var values = new double[length];

        for (var i = 0; i < length; i += 2)
        {
            // Guard the log against an exact zero draw.
            var u1 = Math.Max(random.NextDouble(), double.Epsilon);
            var u2 = random.NextDouble();
            var radius = sigma * Math.Sqrt(-2d * Math.Log(u1));
            var angle = 2d * Math.PI * u2;

            values[i] = radius * Math.Cos(angle);

            if (i + 1 < length)
            {
                values[i + 1] = radius * Math.Sin(angle);
            }
        }

        return values;
    }

    /// <summary>
    /// An MA(1): <c>y_t = e_t + theta * e_(t-1)</c>. Its autocorrelation is
    /// <c>rho_1 = theta / (1 + theta^2)</c> and exactly zero beyond lag 1 — another
    /// identity that needs no reference implementation to check against.
    /// </summary>
    public static double[] Ma1(int length, double theta, int seed, double sigma = 1d)
    {
        var innovations = Gaussian(length + 1, seed, sigma);
        var series = new double[length];

        for (var i = 0; i < length; i++)
        {
            series[i] = innovations[i + 1] + (theta * innovations[i]);
        }

        return series;
    }
}
