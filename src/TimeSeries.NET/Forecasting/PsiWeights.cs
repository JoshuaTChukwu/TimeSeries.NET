using TimeSeries.Transforms;

namespace TimeSeries.Forecasting;

/// <summary>
/// The moving-average representation of a fitted model: the weights <c>psi_j</c> in
/// <c>y_(t+h) - y-hat_(t+h) = sum_(j&lt;h) psi_j e_(t+h-j)</c>, from which every
/// forecast-error variance follows.
/// </summary>
public static class PsiWeights
{
    /// <summary>
    /// ψ-weights of an ARMA(p, q) on its own scale:
    /// <c>psi_0 = 1</c>, <c>psi_j = theta_j + sum_k phi_k psi_(j-k)</c>.
    /// </summary>
    /// <param name="phi">Autoregressive coefficients <c>phi_1 .. phi_p</c>.</param>
    /// <param name="theta">Moving-average coefficients <c>theta_1 .. theta_q</c>, additive convention.</param>
    /// <param name="count">How many weights to produce, <c>psi_0 .. psi_(count-1)</c>.</param>
    /// <returns>The weights.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative.</exception>
    public static double[] Arma(ReadOnlySpan<double> phi, ReadOnlySpan<double> theta, int count)
    {
        if (count < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, "Count must be zero or greater.");
        }

        var psi = new double[count];

        if (count == 0)
        {
            return psi;
        }

        psi[0] = 1d;

        for (var j = 1; j < count; j++)
        {
            var value = j <= theta.Length ? theta[j - 1] : 0d;
            var depth = Math.Min(j, phi.Length);

            for (var k = 1; k <= depth; k++)
            {
                value += phi[k - 1] * psi[j - k];
            }

            psi[j] = value;
        }

        return psi;
    }

    /// <summary>
    /// Carries ψ-weights through the differencing operator, giving the weights on the
    /// original scale: one running sum per non-seasonal difference, one lag-<c>s</c>
    /// running sum per seasonal difference.
    /// </summary>
    /// <param name="psi">Weights on the differenced scale.</param>
    /// <param name="spec">The differencing to undo.</param>
    /// <returns>Weights of the same length on the original scale.</returns>
    public static double[] Integrate(ReadOnlySpan<double> psi, DifferenceSpec spec)
    {
        var weights = psi.ToArray();

        for (var i = 0; i < spec.SeasonalOrder; i++)
        {
            for (var j = spec.Period; j < weights.Length; j++)
            {
                weights[j] += weights[j - spec.Period];
            }
        }

        for (var i = 0; i < spec.Order; i++)
        {
            for (var j = 1; j < weights.Length; j++)
            {
                weights[j] += weights[j - 1];
            }
        }

        return weights;
    }

    /// <summary>
    /// Forecast-error standard deviations for steps <c>1 .. weights.Length</c>:
    /// <c>sigma sqrt(sum_(j&lt;h) psi_j^2)</c>.
    /// </summary>
    /// <param name="weights">ψ-weights on the scale being forecast.</param>
    /// <param name="innovationVariance"><c>sigma^2</c>.</param>
    /// <returns>One standard error per step.</returns>
    public static double[] StandardErrors(ReadOnlySpan<double> weights, double innovationVariance)
    {
        var errors = new double[weights.Length];
        var cumulative = 0d;

        for (var h = 0; h < weights.Length; h++)
        {
            cumulative += weights[h] * weights[h];
            errors[h] = Math.Sqrt(innovationVariance * cumulative);
        }

        return errors;
    }
}
