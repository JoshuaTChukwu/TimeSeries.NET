using MongoDB.Bson;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Data.MongoDb.Tests;

public class MongoTimeSeriesSourceTests
{
    private static readonly ArimaOptions Options = new() { Order = new(1, 1, 1), PilotOrder = 6 };

    /// <summary>A fresh driver cursor per open, as the real driver gives; the last one is returned for inspection.</summary>
    private static (MongoTimeSeriesSource Source, CursorTracker Cursor) Make(
        IReadOnlyList<BsonDocument> documents, MongoSeriesQuery? query = null, DbSourceOptions? options = null, int driverBatch = 1000)
    {
        var tracker = new CursorTracker();
        var source = new MongoTimeSeriesSource(query ?? new MongoSeriesQuery(), options, _ =>
        {
            tracker.Last = new FakeAsyncCursor(documents, driverBatch);
            return Task.FromResult<MongoDB.Driver.IAsyncCursor<BsonDocument>>(tracker.Last);
        });
        return (source, tracker);
    }

    private sealed class CursorTracker
    {
        public FakeAsyncCursor? Last { get; set; }

        public bool Disposed => Last?.Disposed ?? false;

        public int Batches => Last?.Batches ?? 0;
    }

    [Theory]
    [InlineData(1000, 8192)]
    [InlineData(101, 700)]
    [InlineData(8192, 333)]
    public async Task FitFromDocuments_EqualsTheInMemoryFit_Exactly(int driverBatch, int estimatorBatch)
    {
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(3_000, [0.5], [0.3], seed: 1), 1);
        var (source, cursor) = Make(Documents.Daily(series), new MongoSeriesQuery { TimeField = "ts", BatchSize = estimatorBatch }, driverBatch: driverBatch);
        var model = new ArimaModel(Options);

        var streamed = await model.FitAsync(source);
        var inMemory = model.Fit(series);

        Assert.Equal(inMemory.AutoRegressive.ToArray(), streamed.AutoRegressive.ToArray());
        Assert.Equal(inMemory.MovingAverage.ToArray(), streamed.MovingAverage.ToArray());
        Assert.Equal(inMemory.Intercept, streamed.Intercept);
        Assert.Equal(series.Length, streamed.ObservationCount);
        Assert.True(cursor.Disposed);
    }

    [Fact]
    public async Task ExogenousFields_AndDecimalOrIntegerValues_AreRead()
    {
        var x1 = SeriesGenerator.Ar1(2_000, 0.4, seed: 2);
        var series = SeriesGenerator.Integrate(SeriesGenerator.Arma(2_000, [0.5], [0.3], seed: 3), 1).Take(2_000).ToArray();
        var documents = Documents.Daily(series, x1: x1);
        documents[10]["value"] = new BsonDecimal128((decimal)series[10]);
        documents[11]["x1"] = new BsonInt32(3);
        x1[11] = 3;

        var (source, _) = Make(documents, new MongoSeriesQuery { ExogenousFields = ["x1"] });
        var model = new ArimaModel(Options);

        var streamed = await model.FitAsync(source);
        var inMemory = model.Fit(series.Select((v, i) => i == 10 ? (double)(decimal)v : v).ToArray(), ExogenousMatrix.FromColumn(x1));

        Assert.Equal(inMemory.ExogenousCoefficients.ToArray(), streamed.ExogenousCoefficients.ToArray());
    }

    [Fact]
    public async Task Gap_Throws_WithBsonDates()
    {
        var (source, _) = Make(
            Documents.Daily(SeriesGenerator.Ar1(400, 0.5, seed: 4), skipIndex: 77),
            new MongoSeriesQuery { TimeField = "ts" },
            new DbSourceOptions { ExpectedStep = TimeSpan.FromDays(1) });

        var exception = await Assert.ThrowsAsync<SeriesGapException>(
            () => new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitAsync(source).AsTask());

        Assert.StartsWith("2020-03-17", exception.Before, StringComparison.Ordinal);
        Assert.StartsWith("2020-03-19", exception.After, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ForwardFill_WithNumericTime_AcrossDriverAndEstimatorBatches()
    {
        var series = SeriesGenerator.Ar1(600, 0.5, seed: 5);
        const int Missing = 250;
        var documents = Documents.Daily(series, numericTime: true).Where(d => d["ts"].AsInt32 is < Missing or > Missing + 2).ToList();

        var filled = (double[])series.Clone();
        for (var i = 0; i < 3; i++)
        {
            filled[Missing + i] = series[Missing - 1];
        }

        var (source, _) = Make(
            documents,
            new MongoSeriesQuery { TimeField = "ts", BatchSize = 251 },
            new DbSourceOptions { Gaps = GapPolicy.ForwardFill, ExpectedNumericStep = 1 },
            driverBatch: 97);

        await SourceContract.VerifyAsync(source, filled, null, 251);
    }

    [Fact]
    public async Task GroupedScan_FitsEachKey_InOneCursor()
    {
        var a = SeriesGenerator.Ar1(1_500, 0.3, seed: 6);
        var b = SeriesGenerator.Ar1(1_500, 0.8, seed: 7);
        var documents = Documents.Daily(a, key: "ACC-001").Concat(Documents.Daily(b, key: "ACC-002")).ToList();
        var (source, cursor) = Make(documents, new MongoSeriesQuery { KeyField = "key", TimeField = "ts", BatchSize = 400 }, driverBatch: 1000);
        var model = new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) });

        var results = new List<KeyedArimaFit>();
        await foreach (var result in model.FitManyAsync(source, FitManyOptions.Sequential))
        {
            results.Add(result);
        }

        Assert.Equal(["ACC-001", "ACC-002"], results.Select(r => r.Key).ToArray());
        Assert.Equal(model.Fit(a).AutoRegressive.ToArray(), results[0].Fit!.AutoRegressive.ToArray());
        Assert.Equal(model.Fit(b).AutoRegressive.ToArray(), results[1].Fit!.AutoRegressive.ToArray());
        Assert.Equal(3, cursor.Batches);
        Assert.True(cursor.Disposed);
    }

    [Fact]
    public async Task KeyOutOfOrder_Throws()
    {
        var documents = Documents.Daily(SeriesGenerator.Ar1(50, 0.5, seed: 8), key: "B")
            .Concat(Documents.Daily(SeriesGenerator.Ar1(50, 0.5, seed: 9), key: "A")).ToList();
        var (source, _) = Make(documents, new MongoSeriesQuery { KeyField = "key" });

        await Assert.ThrowsAsync<SeriesOrderException>(async () =>
        {
            await foreach (var _ in new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitManyAsync(source))
            {
            }
        });
    }

    [Fact]
    public async Task MissingOrNullValue_ThrowsNamingTheDocument()
    {
        var documents = Documents.Daily(SeriesGenerator.Ar1(300, 0.5, seed: 10));
        documents[42].Remove("value");
        var (source, _) = Make(documents);

        var exception = await Assert.ThrowsAsync<InvalidSeriesException>(
            () => new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitAsync(source).AsTask());

        Assert.Equal(42, exception.Index);
        Assert.Equal("value", exception.ParamName);
    }

    [Fact]
    public void QueryValidation_AndDefaultSort()
    {
        Assert.Throws<ArgumentException>(() => new MongoSeriesQuery { ValueField = " " }.Validate());
        Assert.Throws<ArgumentException>(() => new MongoSeriesQuery { ExogenousFields = ["value"] }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new MongoSeriesQuery { BatchSize = 0 }.Validate());

        Assert.Null(new MongoSeriesQuery().EffectiveSort());
        Assert.NotNull(new MongoSeriesQuery { TimeField = "ts" }.EffectiveSort());
        Assert.NotNull(new MongoSeriesQuery { KeyField = "k", TimeField = "ts" }.EffectiveSort());

        Assert.Throws<ArgumentException>(() => Make([], new MongoSeriesQuery { TimeField = "ts" }, new DbSourceOptions { Gaps = GapPolicy.Interpolate }));
    }

    [Fact]
    public async Task KeyedQuery_UsedAsASingleSeries_IsRefused()
    {
        var (source, _) = Make(Documents.Daily(SeriesGenerator.Ar1(50, 0.5, seed: 11), key: "K"), new MongoSeriesQuery { KeyField = "key" });

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => new ArimaModel(new ArimaOptions { Order = new(1, 0, 0) }).FitAsync(source).AsTask());
    }
}
