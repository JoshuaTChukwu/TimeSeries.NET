using Microsoft.Extensions.DependencyInjection;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Extensions.DependencyInjection.Tests;

public class RegistrationTests
{
    [Fact]
    public void AddTimeSeries_RegistersASingletonFactory_Idempotently()
    {
        var services = new ServiceCollection().AddTimeSeries().AddTimeSeries();
        using var provider = services.BuildServiceProvider();

        var a = provider.GetRequiredService<IArimaModelFactory>();
        var b = provider.GetRequiredService<IArimaModelFactory>();

        Assert.Same(a, b);
        Assert.Single(services, d => d.ServiceType == typeof(IArimaModelFactory));
    }

    [Fact]
    public void AddArimaModel_RegistersAUsableSingleton()
    {
        var options = new ArimaOptions { Order = new(1, 0, 0) };
        using var provider = new ServiceCollection().AddArimaModel(options).BuildServiceProvider();

        var model = provider.GetRequiredService<ArimaModel>();
        var fit = model.Fit(SeriesGenerator.Ar1(2_000, 0.6, seed: 1));

        Assert.Same(model, provider.GetRequiredService<ArimaModel>());
        Assert.Equal(options, model.Options);
        Assert.Equal(0.6, fit.AutoRegressive.Span[0], 0.05);
    }

    [Fact]
    public void AddAutoArima_RegistersAUsableSingleton()
    {
        var options = new AutoArimaOptions { MaxP = 2, MaxQ = 1, MaxPilotOrder = 8 };
        using var provider = new ServiceCollection().AddAutoArima(options).BuildServiceProvider();

        var auto = provider.GetRequiredService<AutoArima>();
        var selection = auto.Select(SeriesGenerator.Ar1(2_000, 0.6, seed: 2));

        Assert.Equal(0, selection.Best.Order.D);
        Assert.NotNull(provider.GetRequiredService<IArimaModelFactory>());
    }

    [Fact]
    public void Factory_CreatesEveryEstimator_AndRestoresState()
    {
        var factory = new ArimaModelFactory();
        var options = new ArimaOptions { Order = new(1, 0, 0) };
        var series = SeriesGenerator.Ar1(1_000, 0.5, seed: 3);

        var incremental = factory.CreateIncremental(options);
        incremental.Fold(series);

        using var state = new MemoryStream();
        incremental.SaveTo(state);
        state.Position = 0;

        var restored = factory.RestoreIncremental(state, options);

        Assert.Equal(incremental.Solve().AutoRegressive.ToArray(), restored.Solve().AutoRegressive.ToArray());
        Assert.Equal(options, factory.Create(options).Options);
        Assert.NotNull(factory.CreateAutoArima(new AutoArimaOptions { MaxP = 1, MaxQ = 1, MaxPilotOrder = 4 }));
    }

    [Fact]
    public void InvalidOptions_FailAtRegistration_NotAtResolve()
    {
        var services = new ServiceCollection();

        Assert.Throws<ArgumentOutOfRangeException>(() => services.AddArimaModel(new ArimaOptions { Order = new(-1, 0, 0) }));
        Assert.Throws<ArgumentNullException>(() => services.AddArimaModel(null!));
        Assert.Throws<ArgumentNullException>(() => ((IServiceCollection)null!).AddTimeSeries());
    }
}
