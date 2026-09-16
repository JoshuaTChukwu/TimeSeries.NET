namespace TimeSeries.Solvers;

/// <summary>
/// The Durbin-Levinson recursion over an autocovariance sequence: partial
/// autocorrelations and Yule-Walker autoregressive fits of every order at once, in
/// <c>O(L^2)</c> with no matrix inversion.
/// </summary>
/// <remarks>
/// <para>
/// The recursion is stable only for a positive semi-definite sequence, which is why the
/// library's autocovariances are estimated with the divide-by-n estimator. Fed a sequence
/// that is not — a divide-by-(n-k) estimate, or entries lifted from a Gram matrix that is
/// not Toeplitz — it can produce partial autocorrelations outside [-1, 1] and a negative
/// innovation variance. Both are detected and reported rather than returned.
/// </para>
/// <para>
/// Step <c>k</c> computes <c>phi_kk</c>, the partial autocorrelation at lag <c>k</c>, and
/// updates the AR(<c>k</c>) coefficients from the AR(<c>k-1</c>) ones:
/// </para>
/// <code>
/// phi_kk  = ( gamma_k - sum_j phi_(k-1,j) gamma_(k-j) ) / v_(k-1)
/// phi_kj  = phi_(k-1,j) - phi_kk phi_(k-1,k-j)          j = 1 .. k-1
/// v_k     = v_(k-1) (1 - phi_kk^2)
/// </code>
/// </remarks>
public static class DurbinLevinson
{
    /// <summary>
    /// Computes partial autocorrelations for lags 1 through <paramref name="maxLag"/>.
    /// </summary>
    /// <param name="gamma">
    /// Autocovariances <c>gamma_0 .. gamma_L</c> with <c>L</c> at least
    /// <paramref name="maxLag"/>. Autocorrelations work equally; the recursion is scale
    /// invariant.
    /// </param>
    /// <param name="maxLag">The deepest lag to compute.</param>
    /// <param name="pacf">
    /// Receives <c>phi_kk</c> for <c>k = 1 .. maxLag</c> at indices <c>0 .. maxLag-1</c>.
    /// Entries past <paramref name="computed"/> are left untouched.
    /// </param>
    /// <param name="computed">
    /// How many lags were computed. Equal to <paramref name="maxLag"/> on success; on a
    /// degenerate sequence, the lag at which the innovation variance reached zero.
    /// </param>
    /// <returns>
    /// True when every requested lag was computed; false when the sequence became
    /// degenerate first — a zero variance, or a partial autocorrelation of magnitude 1,
    /// meaning the process is perfectly predictable from that order.
    /// </returns>
    /// <exception cref="ArgumentException">The spans are inconsistently sized.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is negative.</exception>
    public static bool TryPacf(ReadOnlySpan<double> gamma, int maxLag, Span<double> pacf, out int computed)
    {
        Validate(gamma, maxLag, pacf, nameof(pacf));

        Span<double> previous = maxLag <= 64 ? stackalloc double[maxLag] : new double[maxLag];
        Span<double> current = maxLag <= 64 ? stackalloc double[maxLag] : new double[maxLag];

        return Recurse(gamma, maxLag, pacf, previous, current, out computed, out _);
    }

    /// <summary>
    /// Computes partial autocorrelations for lags 1 through <paramref name="maxLag"/>.
    /// </summary>
    /// <param name="gamma">Autocovariances <c>gamma_0 .. gamma_L</c>, <c>L</c> at least <paramref name="maxLag"/>.</param>
    /// <param name="maxLag">The deepest lag to compute.</param>
    /// <returns><c>phi_kk</c> for <c>k = 1 .. maxLag</c>.</returns>
    /// <exception cref="InvalidOperationException">
    /// The sequence is degenerate before <paramref name="maxLag"/>; the message names the lag.
    /// </exception>
    public static double[] Pacf(ReadOnlySpan<double> gamma, int maxLag)
    {
        var pacf = new double[maxLag];

        if (!TryPacf(gamma, maxLag, pacf, out var computed))
        {
            throw new InvalidOperationException(
                $"The autocovariance sequence is degenerate at lag {computed}: the process is " +
                $"perfectly predictable from order {computed}, so no partial autocorrelation " +
                $"exists beyond it. {maxLag} lags were requested.");
        }

        return pacf;
    }

    /// <summary>
    /// Fits AR(<paramref name="order"/>) by Yule-Walker from the autocovariances.
    /// </summary>
    /// <param name="gamma">Autocovariances <c>gamma_0 .. gamma_L</c>, <c>L</c> at least <paramref name="order"/>.</param>
    /// <param name="order">The autoregressive order <c>p</c>.</param>
    /// <param name="coefficients">Receives <c>phi_1 .. phi_p</c>.</param>
    /// <param name="innovationVariance">
    /// Receives <c>v_p = gamma_0 prod (1 - phi_kk^2)</c>, the one-step prediction error
    /// variance of the fitted model.
    /// </param>
    /// <returns>
    /// True on success; false when the sequence is degenerate before <paramref name="order"/>.
    /// </returns>
    /// <exception cref="ArgumentException">The spans are inconsistently sized.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> is negative.</exception>
    public static bool TryYuleWalker(
        ReadOnlySpan<double> gamma, int order, Span<double> coefficients, out double innovationVariance)
    {
        Validate(gamma, order, coefficients, nameof(coefficients));

        Span<double> pacf = order <= 64 ? stackalloc double[order] : new double[order];
        Span<double> previous = order <= 64 ? stackalloc double[order] : new double[order];
        Span<double> current = order <= 64 ? stackalloc double[order] : new double[order];

        var complete = Recurse(gamma, order, pacf, previous, current, out var computed, out innovationVariance);

        if (complete)
        {
            // After the final step the freshest coefficients sit in `previous`.
            previous.Slice(0, order).CopyTo(coefficients);
        }
        else
        {
            coefficients.Slice(0, order).Clear();
            _ = computed;
        }

        return complete;
    }

    private static void Validate(ReadOnlySpan<double> gamma, int lags, Span<double> output, string outputName)
    {
        if (lags < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(lags), lags, "Lag count must be zero or greater.");
        }

        if (gamma.Length < lags + 1)
        {
            throw new ArgumentException(
                $"{lags} lags need autocovariances gamma_0 through gamma_{lags}, {lags + 1} values; " +
                $"{gamma.Length} supplied.",
                nameof(gamma));
        }

        if (output.Length < lags)
        {
            throw new ArgumentException(
                $"Output must hold at least {lags} values; {output.Length} available.", outputName);
        }
    }

    /// <summary>
    /// The shared recursion. On return <paramref name="previous"/> holds the
    /// AR(<paramref name="computed"/>) coefficients.
    /// </summary>
    private static bool Recurse(
        ReadOnlySpan<double> gamma,
        int maxLag,
        Span<double> pacf,
        Span<double> previous,
        Span<double> current,
        out int computed,
        out double innovationVariance)
    {
        var variance = gamma[0];
        computed = 0;

        if (!(variance > 0d))
        {
            innovationVariance = variance;
            return maxLag == 0;
        }

        for (var k = 1; k <= maxLag; k++)
        {
            var numerator = gamma[k];

            for (var j = 1; j < k; j++)
            {
                numerator -= previous[j - 1] * gamma[k - j];
            }

            var phi = numerator / variance;
            pacf[k - 1] = phi;
            computed = k;

            for (var j = 1; j < k; j++)
            {
                current[j - 1] = previous[j - 1] - (phi * previous[k - j - 1]);
            }

            current[k - 1] = phi;
            variance *= 1d - (phi * phi);

            // The two buffers swap roles each step. Copying is cheaper than juggling
            // references through a Span-returning helper and the arrays are tiny.
            current.Slice(0, k).CopyTo(previous);

            if (Math.Abs(phi) >= 1d || !(variance > 0d))
            {
                innovationVariance = Math.Max(variance, 0d);
                return false;
            }
        }

        innovationVariance = variance;
        return true;
    }
}
