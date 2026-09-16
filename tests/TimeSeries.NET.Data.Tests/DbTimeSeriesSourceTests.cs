using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Data.Tests;

public sealed class DbTimeSeriesSourceTests : IDisposable
{
    private static readonly ArimaOptions Options = new() { Order = new(1, 1, 1), PilotOrder = 6 };
    private readonly SqliteDatabase _db = new();

    public void Dispose() => _db.Dispose();

    private DbTimeSeriesSource Source(string table, DbSeriesQuery? query = null, DbSourceOptions? options = null, bool tracked = false)
    {
        query ??= new DbSeriesQuery { CommandText = $"SELECT ts, value FROM {table} ORDER BY id", ValueColumn = "value" };
        return new DbTimeSeriesSource(tracked ? _db.ConnectTracked : _db.Connect, query, options);
    }

    [Fact]
    public async Task FitFromTheDatabase_EqualsTheInMemoryFit_Exactly()
    {
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(3_000, [0.5], [0.3], seed: 1), 1);
        _db.CreateSeriesTable("prices", SqliteDatabase.Daily(series));
        var model = new ArimaModel(Options);

        var streamed = await model.FitAsync(Source("prices", new DbSeriesQuery
        {
            CommandText = "SELECT ts, value FROM prices ORDER BY id",
            BatchSize = 700,
        }));

        AssertSameFit(model.Fit(series), streamed);
    }

    [Fact]
    public async Task ExogenousColumns_AreReadInQueryOrder_AndTheFitMatches()
    {
        var x1 = SeriesGenerator.Ar1(3_000, 0.4, seed: 2);
        var x2 = SeriesGenerator.Gaussian(3_000, seed: 3);
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(3_000, [0.5], [0.3], seed: 4), 1).Take(3_000).ToArray();
        _db.CreateSeriesTable("bars", SqliteDatabase.Daily(series, x1: x1, x2: x2));
        var model = new ArimaModel(Options);

        // Columns deliberately selected in an order different from the roles.
        var source = Source("bars", new DbSeriesQuery
        {
            CommandText = "SELECT x2, value, ts, x1 FROM bars ORDER BY id",
            ExogenousColumns = ["x1", "x2"],
            TimeColumn = "ts",
        });

        var streamed = await model.FitAsync(source);
        var inMemory = model.Fit(series, ExogenousMatrix.FromColumns(x1, x2));

        AssertSameFit(inMemory, streamed);
        Assert.Equal(inMemory.ExogenousCoefficients.ToArray(), streamed.ExogenousCoefficients.ToArray());
    }

    [Fact]
    public async Task ParametersAreBound()
    {
        var series = SeriesGenerator.Ar1(2_000, 0.5, seed: 5);
        _db.CreateSeriesTable("filtered", SqliteDatabase.Daily(series));

        var source = Source("filtered", new DbSeriesQuery
        {
            CommandText = "SELECT ts, value FROM filtered WHERE t >= @from ORDER BY id",
            Parameters = { ["@from"] = 500L },
        });

        var fit = await new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitAsync(source);

        Assert.Equal(1_500, fit.ObservationCount);
    }

    [Fact]
    public async Task Gap_Throws_NamingTheTimestampsEitherSide()
    {
        var series = SeriesGenerator.Ar1(500, 0.5, seed: 6);
        _db.CreateSeriesTable("gappy", SqliteDatabase.Daily(series, skipIndex: 137));

        var source = Source("gappy", new DbSeriesQuery
        {
            CommandText = "SELECT ts, value FROM gappy ORDER BY id",
            TimeColumn = "ts",
        }, new DbSourceOptions { ExpectedStep = TimeSpan.FromDays(1) });

        var exception = await Assert.ThrowsAsync<SeriesGapException>(
            () => new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitAsync(source).AsTask());

        Assert.StartsWith("2020-05-16", exception.Before, StringComparison.Ordinal);
        Assert.StartsWith("2020-05-18", exception.After, StringComparison.Ordinal);
        Assert.Contains("1 observation(s) are missing", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task MaxGap_LetsExpectedIrregularityThrough()
    {
        var series = SeriesGenerator.Ar1(500, 0.5, seed: 7);
        _db.CreateSeriesTable("weekend", SqliteDatabase.Daily(series, skipIndex: 200));

        var source = Source("weekend", new DbSeriesQuery
        {
            CommandText = "SELECT ts, value FROM weekend ORDER BY id",
            TimeColumn = "ts",
        }, new DbSourceOptions { ExpectedStep = TimeSpan.FromDays(1), MaxGap = TimeSpan.FromDays(4) });

        var fit = await new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitAsync(source);

        Assert.Equal(499, fit.ObservationCount);
    }

    [Theory]
    [InlineData(GapPolicy.ForwardFill)]
    [InlineData(GapPolicy.Interpolate)]
    public async Task FillPolicies_EqualFittingTheFilledArray(GapPolicy policy)
    {
        var series = SeriesGenerator.Ar1(600, 0.5, seed: 8);
        var x1 = SeriesGenerator.Uniform(600, seed: 9);
        const int Missing = 250;

        // Remove three consecutive rows, so the fill has to produce three observations.
        var rows = SqliteDatabase.Daily(series, x1: x1).Where(r => r.NumericTime is < Missing or > Missing + 2);
        _db.CreateSeriesTable("holes", rows);

        // The array the source should reconstruct.
        var filled = (double[])series.Clone();
        var filledX = (double[])x1.Clone();
        for (var i = 1; i <= 3; i++)
        {
            var fraction = i / 4d;
            filled[Missing - 1 + i] = policy == GapPolicy.ForwardFill
                ? series[Missing - 1]
                : series[Missing - 1] + ((series[Missing + 3] - series[Missing - 1]) * fraction);
            filledX[Missing - 1 + i] = policy == GapPolicy.ForwardFill
                ? x1[Missing - 1]
                : x1[Missing - 1] + ((x1[Missing + 3] - x1[Missing - 1]) * fraction);
        }

        var source = Source("holes", new DbSeriesQuery
        {
            CommandText = "SELECT t, value, x1 FROM holes ORDER BY id",
            TimeColumn = "t",
            ExogenousColumns = ["x1"],
            BatchSize = 251, // the fill straddles a batch boundary
        }, new DbSourceOptions { Gaps = policy, ExpectedNumericStep = 1 });

        await SourceContract.VerifyAsync(source, filled, ExogenousMatrix.FromColumn(filledX), 251);
    }

    [Fact]
    public async Task OutOfOrderTimestamps_Throw()
    {
        var series = SeriesGenerator.Ar1(100, 0.5, seed: 10);
        var rows = SqliteDatabase.Daily(series).ToList();
        (rows[40], rows[41]) = (rows[41], rows[40]);
        _db.CreateSeriesTable("shuffled", rows);

        // MaxGap is widened so the two-day jump created by the swap is not reported as a
        // gap first; what must be caught is the timestamp that then goes backwards.
        var source = Source("shuffled", new DbSeriesQuery
        {
            CommandText = "SELECT ts, value FROM shuffled ORDER BY id",
            TimeColumn = "ts",
        }, new DbSourceOptions { ExpectedStep = TimeSpan.FromDays(1), MaxGap = TimeSpan.FromDays(10) });

        await Assert.ThrowsAsync<SeriesOrderException>(
            () => new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitAsync(source).AsTask());
    }

    [Fact]
    public async Task NullValue_ThrowsNamingTheRow()
    {
        var series = SeriesGenerator.Ar1(300, 0.5, seed: 11);
        var rows = SqliteDatabase.Daily(series).ToList();
        rows[123] = rows[123] with { Value = null };
        _db.CreateSeriesTable("nulls", rows);

        var exception = await Assert.ThrowsAsync<InvalidSeriesException>(
            () => new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitAsync(Source("nulls")).AsTask());

        Assert.Equal(123, exception.Index);
        Assert.Equal("value", exception.ParamName);
    }

    [Fact]
    public async Task GroupedScan_FitsEachKey_AndMatchesPerKeyFits()
    {
        var a = SeriesGenerator.Ar1(2_000, 0.3, seed: 12);
        var b = SeriesGenerator.Ar1(15, 0.5, seed: 13);
        var c = SeriesGenerator.Ar1(2_000, 0.8, seed: 14);
        var rows = SqliteDatabase.Daily(a, key: "ACC-001")
            .Concat(SqliteDatabase.Daily(b, key: "ACC-002"))
            .Concat(SqliteDatabase.Daily(c, key: "ACC-003"));
        _db.CreateSeriesTable("accounts", rows);

        var source = Source("accounts", new DbSeriesQuery
        {
            CommandText = "SELECT key, ts, value FROM accounts ORDER BY key, ts",
            KeyColumn = "key",
            TimeColumn = "ts",
            BatchSize = 300,
        }, new DbSourceOptions { ExpectedStep = TimeSpan.FromDays(1) });
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) });

        var results = new List<KeyedArimaFit>();
        await foreach (var result in model.FitManyAsync(source))
        {
            results.Add(result);
        }

        Assert.Equal(["ACC-001", "ACC-002", "ACC-003"], results.Select(r => r.Key).ToArray());
        AssertSameFit(model.Fit(a), results[0].Fit!);
        Assert.IsType<InsufficientDataException>(results[1].Error);
        Assert.Equal(15, results[1].ObservationCount);
        AssertSameFit(model.Fit(c), results[2].Fit!);
    }

    [Fact]
    public async Task GroupedScan_KeyOutOfOrder_Throws()
    {
        var rows = SqliteDatabase.Daily(SeriesGenerator.Ar1(50, 0.5, seed: 15), key: "B")
            .Concat(SqliteDatabase.Daily(SeriesGenerator.Ar1(50, 0.5, seed: 16), key: "A"));
        _db.CreateSeriesTable("unordered", rows);

        var source = Source("unordered", new DbSeriesQuery
        {
            CommandText = "SELECT key, value FROM unordered ORDER BY id",
            KeyColumn = "key",
        });

        await Assert.ThrowsAsync<SeriesOrderException>(async () =>
        {
            await foreach (var _ in new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitManyAsync(source))
            {
            }
        });
    }

    [Fact]
    public async Task KeyedQuery_UsedAsASingleSeries_IsRefused()
    {
        _db.CreateSeriesTable("keyed", SqliteDatabase.Daily(SeriesGenerator.Ar1(50, 0.5, seed: 17), key: "K"));
        var source = Source("keyed", new DbSeriesQuery { CommandText = "SELECT key, value FROM keyed", KeyColumn = "key" });

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitAsync(source).AsTask());
    }

    [Fact]
    public async Task TheConnectionIsDisposed_AfterAFit_AndAfterACancelledFit()
    {
        var series = SeriesGenerator.Ar1(5_000, 0.5, seed: 18);
        _db.CreateSeriesTable("conn", SqliteDatabase.Daily(series));
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) });

        await model.FitAsync(Source("conn", tracked: true));
        Assert.Single(_db.Issued);
        Assert.Equal(System.Data.ConnectionState.Closed, _db.Issued[0].State);

        using var cts = new CancellationTokenSource();
        var counting = new CountingSource(Source("conn", new DbSeriesQuery
        {
            CommandText = "SELECT ts, value FROM conn ORDER BY id",
            BatchSize = 500,
        }, tracked: true), cts, cancelAfter: 3);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => model.FitAsync(counting, cts.Token).AsTask());
        Assert.Equal(2, _db.Issued.Count);
        Assert.Equal(System.Data.ConnectionState.Closed, _db.Issued[1].State);
    }

    [Fact]
    public async Task SatisfiesTheSourceContract()
    {
        var series = SeriesGenerator.Uniform(1_234, seed: 19);
        var x1 = SeriesGenerator.Uniform(1_234, seed: 20);
        _db.CreateSeriesTable("contract", SqliteDatabase.Daily(series, x1: x1));

        var source = Source("contract", new DbSeriesQuery
        {
            CommandText = "SELECT ts, value, x1 FROM contract ORDER BY id",
            ExogenousColumns = ["x1"],
            BatchSize = 100,
        });

        await SourceContract.VerifyAsync(source, series, ExogenousMatrix.FromColumn(x1), 100);
    }

    [Fact]
    public void Construction_Validation()
    {
        Assert.Throws<ArgumentNullException>(() => new DbTimeSeriesSource(null!, new DbSeriesQuery { CommandText = "x" }));
        Assert.Throws<ArgumentNullException>(() => new DbTimeSeriesSource(_db.Connect, null!));
        Assert.Throws<ArgumentException>(() => new DbTimeSeriesSource(_db.Connect, new DbSeriesQuery()));
        Assert.Throws<ArgumentException>(() => new DbTimeSeriesSource(_db.Connect, new DbSeriesQuery
        {
            CommandText = "x",
            ExogenousColumns = ["value"],
        }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DbTimeSeriesSource(_db.Connect, new DbSeriesQuery { CommandText = "x", BatchSize = 0 }));

        // A fill policy without a step cannot know how many rows to fill.
        Assert.Throws<ArgumentException>(() => new DbTimeSeriesSource(
            _db.Connect,
            new DbSeriesQuery { CommandText = "x", TimeColumn = "ts" },
            new DbSourceOptions { Gaps = GapPolicy.ForwardFill }));
    }

    private static void AssertSameFit(ArimaFit expected, ArimaFit actual)
    {
        Assert.Equal(expected.AutoRegressive.ToArray(), actual.AutoRegressive.ToArray());
        Assert.Equal(expected.MovingAverage.ToArray(), actual.MovingAverage.ToArray());
        Assert.Equal(expected.Intercept, actual.Intercept);
        Assert.Equal(expected.InnovationVariance, actual.InnovationVariance);
        Assert.Equal(expected.ObservationCount, actual.ObservationCount);
    }
}
