namespace TimeSeries.Statistics;

/// <summary>
/// The standard normal distribution, to double precision and without a dependency.
/// </summary>
/// <remarks>
/// The quantile is Wichura's algorithm AS 241 (PPND16), accurate to about 1e-16 across
/// (0, 1). The distribution function is Marsaglia's Taylor-series evaluation in the body
/// and the Mills-ratio asymptotic in the far tails. Prediction intervals need the
/// quantile; the distribution function is here so the two round-trip in tests and so
/// callers can turn a statistic into a p-value.
/// </remarks>
public static class NormalDistribution
{
    private const double LogSqrtTwoPi = 0.91893853320467274178;

    /// <summary>
    /// The quantile function <c>Phi^-1(p)</c>.
    /// </summary>
    /// <param name="p">A probability strictly inside (0, 1).</param>
    /// <returns>The <c>z</c> with <c>P(Z &lt;= z) = p</c>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="p"/> is outside (0, 1).</exception>
    public static double Quantile(double p)
    {
        if (!(p > 0d && p < 1d))
        {
            throw new ArgumentOutOfRangeException(
                nameof(p), p, "A quantile is defined only for probabilities strictly inside (0, 1).");
        }

        var q = p - 0.5;

        if (Math.Abs(q) <= 0.425)
        {
            var r = 0.180625 - (q * q);

            return q * (((((((2.5090809287301226727e+3 * r + 3.3430575583588128105e+4) * r
                + 6.7265770927008700853e+4) * r + 4.5921953931549871457e+4) * r
                + 1.3731693765509461125e+4) * r + 1.9715909503065514427e+3) * r
                + 1.3314166789178437745e+2) * r + 3.3871328727963666080e+0)
                / (((((((5.2264952788528545610e+3 * r + 2.8729085735721942674e+4) * r
                + 3.9307895800092710610e+4) * r + 2.1213794301586595867e+4) * r
                + 5.3941960214247511077e+3) * r + 6.8718700749205790830e+2) * r
                + 4.2313330701600911252e+1) * r + 1d);
        }

        var tail = q < 0d ? p : 1d - p;
        var s = Math.Sqrt(-Math.Log(tail));
        double value;

        if (s <= 5d)
        {
            s -= 1.6;
            value = (((((((7.74545014278341407640e-4 * s + 2.27238449892691845833e-2) * s
                + 2.41780725177450611770e-1) * s + 1.27045825245236838258e+0) * s
                + 3.64784832476320460504e+0) * s + 5.76949722146069140550e+0) * s
                + 4.63033784615654529590e+0) * s + 1.42343711074968357734e+0)
                / (((((((1.05075007164441684324e-9 * s + 5.47593808499534494600e-4) * s
                + 1.51986665636164571966e-2) * s + 1.48103976427480074590e-1) * s
                + 6.89767334985100004550e-1) * s + 1.67638483018380384940e+0) * s
                + 2.05319162663775882187e+0) * s + 1d);
        }
        else
        {
            s -= 5d;
            value = (((((((2.01033439929228813265e-7 * s + 2.71155556874348757815e-5) * s
                + 1.24266094738807843860e-3) * s + 2.65321895265761230930e-2) * s
                + 2.96560571828504891230e-1) * s + 1.78482653991729133580e+0) * s
                + 5.46378491116411436990e+0) * s + 6.65790464350110377720e+0)
                / (((((((2.04426310338993978564e-15 * s + 1.42151175831644588870e-7) * s
                + 1.84631831751005468180e-5) * s + 7.86869131145613259100e-4) * s
                + 1.48753612908506148525e-2) * s + 1.36929880922735805310e-1) * s
                + 5.99832206555887937690e-1) * s + 1d);
        }

        return q < 0d ? -value : value;
    }

    /// <summary>
    /// The distribution function <c>Phi(x) = P(Z &lt;= x)</c>.
    /// </summary>
    /// <param name="x">Any real value.</param>
    /// <returns>The probability, in [0, 1].</returns>
    public static double Cdf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x > 7d)
        {
            return 1d - UpperTail(x);
        }

        if (x < -7d)
        {
            return UpperTail(-x);
        }

        // Marsaglia (2004): Phi(x) = 1/2 + phi(x) (x + x^3/3 + x^5/(3*5) + ...).
        var sum = x;
        var term = x;
        var previous = 0d;
        var square = x * x;
        var i = 1;

        while (sum != previous)
        {
            previous = sum;
            i += 2;
            term *= square / i;
            sum = previous + term;
        }

        return 0.5 + (sum * Math.Exp((-0.5 * square) - LogSqrtTwoPi));
    }

    /// <summary>
    /// The two-sided critical value for a confidence level: <c>Phi^-1(1 - (1 - confidence) / 2)</c>.
    /// </summary>
    /// <param name="confidence">A confidence level strictly inside (0, 1), such as 0.95.</param>
    /// <returns>The <c>z</c> such that <c>P(|Z| &lt;= z) = confidence</c>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="confidence"/> is outside (0, 1).</exception>
    public static double TwoSidedCriticalValue(double confidence)
    {
        if (!(confidence > 0d && confidence < 1d))
        {
            throw new ArgumentOutOfRangeException(
                nameof(confidence), confidence, "Confidence must lie strictly inside (0, 1).");
        }

        return Quantile(1d - ((1d - confidence) / 2d));
    }

    /// <summary>
    /// <c>1 - Phi(x)</c> for <c>x &gt; 7</c> by the Mills-ratio asymptotic series. The
    /// omitted term is below 1e-6 of a value that is itself below 1e-11, so the result is
    /// correct to well past double precision in absolute terms.
    /// </summary>
    private static double UpperTail(double x)
    {
        var square = x * x;
        var series = 1d - (1d / square) + (3d / (square * square))
            - (15d / (square * square * square))
            + (105d / (square * square * square * square));

        return Math.Exp((-0.5 * square) - LogSqrtTwoPi) / x * series;
    }
}
