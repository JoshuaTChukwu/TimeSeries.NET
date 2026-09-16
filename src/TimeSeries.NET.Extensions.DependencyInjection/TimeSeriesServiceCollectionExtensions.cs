using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace TimeSeries.Extensions.DependencyInjection;

/// <summary>Registration helpers for <see cref="IServiceCollection"/>.</summary>
public static class TimeSeriesServiceCollectionExtensions
{
    /// <summary>
    /// Registers <see cref="IArimaModelFactory"/> as a singleton. Idempotent.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns><paramref name="services"/>, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services"/> is null.</exception>
    public static IServiceCollection AddTimeSeries(this IServiceCollection services)
    {
        if (services is null)
        {
            throw new ArgumentNullException(nameof(services));
        }

        services.TryAddSingleton<IArimaModelFactory, ArimaModelFactory>();
        return services;
    }

    /// <summary>
    /// Registers a singleton <see cref="ArimaModel"/> for a fixed order. The estimator is
    /// stateless and thread-safe, so one instance serves the whole application.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <param name="options">The model. Validated now, not at first resolve.</param>
    /// <returns><paramref name="services"/>, for chaining.</returns>
    /// <exception cref="ArgumentNullException">An argument is null.</exception>
    public static IServiceCollection AddArimaModel(this IServiceCollection services, ArimaOptions options)
    {
        if (services is null)
        {
            throw new ArgumentNullException(nameof(services));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        var model = new ArimaModel(options);
        services.AddTimeSeries();
        services.AddSingleton(model);
        return services;
    }

    /// <summary>
    /// Registers a singleton <see cref="AutoArima"/> for a search space.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <param name="options">The search space. Validated now.</param>
    /// <returns><paramref name="services"/>, for chaining.</returns>
    /// <exception cref="ArgumentNullException">An argument is null.</exception>
    public static IServiceCollection AddAutoArima(this IServiceCollection services, AutoArimaOptions options)
    {
        if (services is null)
        {
            throw new ArgumentNullException(nameof(services));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        var selector = new AutoArima(options);
        services.AddTimeSeries();
        services.AddSingleton(selector);
        return services;
    }
}
