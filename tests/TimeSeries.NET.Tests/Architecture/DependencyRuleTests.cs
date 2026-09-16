using System.Reflection;
using TimeSeries.Transforms;

namespace TimeSeries.Tests.Architecture;

public class DependencyRuleTests
{
    [Fact]
    public void TheCoreAssemblyReferencesNothingOutsideTheBaseClassLibrary()
    {
        // The zero-dependency promise is on the front of the README, so it is asserted
        // rather than reviewed for. A well-meaning package reference fails the build here
        // instead of quietly ending the claim.
        //
        // Scoped to the net8.0 build, which is what the test project loads. The
        // netstandard2.0 leg carries System.Memory by design (Decision 1) — that one
        // still satisfies the System.* rule, but Microsoft.Bcl.AsyncInterfaces arrives
        // with the async source interfaces at M6 and will not, so this list will need a
        // per-framework allowance at that point rather than a quiet deletion.
        var core = typeof(DifferenceSpec).Assembly;

        var offenders = core.GetReferencedAssemblies()
            .Select(reference => reference.Name ?? string.Empty)
            .Where(name => !name.Equals("System", StringComparison.Ordinal)
                        && !name.StartsWith("System.", StringComparison.Ordinal)
                        && !name.Equals("netstandard", StringComparison.Ordinal)
                        && !name.Equals("mscorlib", StringComparison.Ordinal))
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

        Assert.True(
            offenders.Length == 0,
            $"The core assembly must reference nothing outside System.*, but references: " +
            $"{string.Join(", ", offenders)}.");
    }

    [Fact]
    public void TheCoreAssemblyIsNamedForThePackage()
    {
        var core = typeof(DifferenceSpec).Assembly;

        Assert.Equal("TimeSeries.NET", core.GetName().Name);
    }

    [Fact]
    public void PublicTypesLiveUnderTheTimeSeriesRootNamespace()
    {
        // The rename exists so that no consumer writing `using TimeSeries.Transforms;`
        // inherits a namespace that shadows System.Math. A stray type under the old root
        // would reintroduce exactly that.
        var core = typeof(DifferenceSpec).Assembly;

        var strays = core.GetExportedTypes()
            .Select(type => type.Namespace ?? string.Empty)
            .Where(ns => !ns.Equals("TimeSeries", StringComparison.Ordinal)
                      && !ns.StartsWith("TimeSeries.", StringComparison.Ordinal))
            .Distinct(StringComparer.Ordinal)
            .OrderBy(ns => ns, StringComparer.Ordinal)
            .ToArray();

        Assert.True(
            strays.Length == 0,
            $"Every public type must sit under the TimeSeries root namespace, but found: " +
            $"{string.Join(", ", strays)}.");
    }

    [Fact]
    public void NoPublicNamespaceShadowsASystemNamespace()
    {
        var shadowed = typeof(DifferenceSpec).Assembly
            .GetExportedTypes()
            .Select(type => type.Namespace ?? string.Empty)
            .Where(ns => ns.Length > 0)
            .Select(ns => ns.Split('.')[^1])
            .Where(leaf => Type.GetType($"System.{leaf}") is not null
                        || AppDomain.CurrentDomain.GetAssemblies()
                            .Any(assembly => assembly.GetType($"System.{leaf}") is not null))
            .Distinct(StringComparer.Ordinal)
            .ToArray();

        Assert.True(
            shadowed.Length == 0,
            $"A namespace leaf that matches a System type name makes every consumer's " +
            $"`using` ambiguous. Offending leaves: {string.Join(", ", shadowed)}.");
    }

    [Fact]
    public void EveryAccumulatorImplementsTheAccumulatorContract()
    {
        // The estimator will depend on Merge and Freeze being available on anything in
        // this folder, so an accumulator that quietly declines the interface is a bug the
        // compiler will not catch until M4.
        var core = typeof(DifferenceSpec).Assembly;

        var accumulators = core.GetExportedTypes()
            .Where(type => type is { IsClass: true, IsAbstract: false }
                        && type.Namespace == "TimeSeries.Accumulators"
                        && type.Name.EndsWith("Accumulator", StringComparison.Ordinal))
            .ToArray();

        Assert.NotEmpty(accumulators);

        foreach (var accumulator in accumulators)
        {
            var implements = accumulator.GetInterfaces().Any(contract =>
                contract.IsGenericType
                && contract.GetGenericTypeDefinition() == typeof(TimeSeries.Accumulators.IAccumulator<,>));

            Assert.True(implements, $"{accumulator.Name} must implement IAccumulator<,>.");
        }
    }

    [Fact]
    public void NothingSurvivingAFitHoldsAReferenceToAnArray()
    {
        // Architecture section 6: nothing that outlives a fit may hold anything
        // proportional to the series length. The accumulator results are the first types
        // to reach that rule, and each is bounded by the model order.
        var bounded = new[]
        {
            typeof(TimeSeries.Accumulators.GramMatrix),
            typeof(TimeSeries.Accumulators.Autocovariances),
            typeof(TimeSeries.Accumulators.Moments),
            typeof(IntegrationState),
        };

        foreach (var type in bounded)
        {
            var mutableFields = type
                .GetFields(BindingFlags.Public | BindingFlags.Instance)
                .Where(field => !field.IsInitOnly)
                .ToArray();

            Assert.True(
                mutableFields.Length == 0,
                $"{type.Name} exposes mutable state: {string.Join(", ", mutableFields.Select(f => f.Name))}.");

            Assert.True(
                type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                    .All(property => !property.CanWrite),
                $"{type.Name} must be immutable once frozen.");
        }
    }
}
