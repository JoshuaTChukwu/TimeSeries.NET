namespace TimeSeries.Statistics;

/// <summary>
/// The chi-squared distribution, to double precision and without a dependency, for
/// turning a portmanteau statistic into a p-value.
/// </summary>
/// <remarks>
/// The survival function is the regularised upper incomplete gamma function
/// <c>Q(k/2, x/2)</c>, evaluated by the series for small arguments and by Lentz's
/// continued fraction otherwise — the standard split, accurate to about 1e-14.
/// </remarks>
public static class ChiSquaredDistribution
{
    private const int MaxIterations = 1000;
    private const double Epsilon = 1e-15;

    /// <summary>
    /// <c>P(X &gt; x)</c> for <c>X ~ chi-squared(k)</c>.
    /// </summary>
    /// <param name="x">The statistic; non-negative.</param>
    /// <param name="degreesOfFreedom">Degrees of freedom <c>k</c>; positive.</param>
    /// <returns>The upper-tail probability in [0, 1].</returns>
    /// <exception cref="ArgumentOutOfRangeException">A negative statistic or non-positive degrees of freedom.</exception>
    public static double SurvivalFunction(double x, double degreesOfFreedom)
    {
        if (!(degreesOfFreedom > 0d))
        {
            throw new ArgumentOutOfRangeException(nameof(degreesOfFreedom), degreesOfFreedom, "Degrees of freedom must be positive.");
        }

        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x < 0d)
        {
            throw new ArgumentOutOfRangeException(nameof(x), x, "A chi-squared statistic cannot be negative.");
        }

        return RegularizedUpperGamma(degreesOfFreedom / 2d, x / 2d);
    }

    /// <summary>
    /// <c>P(X &lt;= x)</c> for <c>X ~ chi-squared(k)</c>.
    /// </summary>
    /// <param name="x">The statistic; non-negative.</param>
    /// <param name="degreesOfFreedom">Degrees of freedom <c>k</c>; positive.</param>
    /// <returns>The lower-tail probability in [0, 1].</returns>
    public static double Cdf(double x, double degreesOfFreedom) => 1d - SurvivalFunction(x, degreesOfFreedom);

    /// <summary>The regularised upper incomplete gamma function <c>Q(a, x)</c>.</summary>
    internal static double RegularizedUpperGamma(double a, double x)
    {
        if (x <= 0d)
        {
            return 1d;
        }

        return x < a + 1d ? 1d - LowerSeries(a, x) : UpperContinuedFraction(a, x);
    }

    /// <summary>Series for <c>P(a, x)</c>, converging fast when <c>x &lt; a + 1</c>.</summary>
    private static double LowerSeries(double a, double x)
    {
        var term = 1d / a;
        var sum = term;
        var ap = a;

        for (var i = 0; i < MaxIterations; i++)
        {
            ap += 1d;
            term *= x / ap;
            sum += term;

            if (Math.Abs(term) < Math.Abs(sum) * Epsilon)
            {
                break;
            }
        }

        return sum * Math.Exp((-x) + (a * Math.Log(x)) - LogGamma(a));
    }

    /// <summary>Lentz's continued fraction for <c>Q(a, x)</c>, converging fast when <c>x &gt;= a + 1</c>.</summary>
    private static double UpperContinuedFraction(double a, double x)
    {
        const double Tiny = 1e-300;
        var b = x + 1d - a;
        var c = 1d / Tiny;
        var d = 1d / b;
        var h = d;

        for (var i = 1; i <= MaxIterations; i++)
        {
            var an = -i * (i - a);
            b += 2d;
            d = (an * d) + b;
            d = Math.Abs(d) < Tiny ? Tiny : d;
            c = b + (an / c);
            c = Math.Abs(c) < Tiny ? Tiny : c;
            d = 1d / d;
            var delta = d * c;
            h *= delta;

            if (Math.Abs(delta - 1d) < Epsilon)
            {
                break;
            }
        }

        return Math.Exp((-x) + (a * Math.Log(x)) - LogGamma(a)) * h;
    }

    /// <summary>Lanczos approximation to <c>ln Gamma(z)</c>, <c>z &gt; 0</c>, accurate to about 1e-15.</summary>
    internal static double LogGamma(double z)
    {
        // Coefficients for g = 7, n = 9 (Godfrey's set).
        ReadOnlySpan<double> coefficients =
        [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        if (z < 0.5)
        {
            // Reflection: Gamma(z) Gamma(1 - z) = pi / sin(pi z).
            return Math.Log(Math.PI / Math.Sin(Math.PI * z)) - LogGamma(1d - z);
        }

        z -= 1d;
        var sum = coefficients[0];
        var t = z + 7.5;

        for (var i = 1; i < coefficients.Length; i++)
        {
            sum += coefficients[i] / (z + i);
        }

        return (0.5 * Math.Log(2d * Math.PI)) + ((z + 0.5) * Math.Log(t)) - t + Math.Log(sum);
    }
}
