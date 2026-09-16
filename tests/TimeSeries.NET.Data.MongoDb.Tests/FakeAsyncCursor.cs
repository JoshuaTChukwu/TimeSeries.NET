using MongoDB.Bson;
using MongoDB.Driver;

namespace TimeSeries.Data.MongoDb.Tests;

/// <summary>
/// The driver's cursor contract over an in-memory list, delivered in batches of a chosen
/// size — including a size that does not line up with the estimator's, so that the
/// enumerator has to survive across estimator batches.
/// </summary>
public sealed class FakeAsyncCursor(IReadOnlyList<BsonDocument> documents, int driverBatchSize) : IAsyncCursor<BsonDocument>
{
    private int _position;

    public int Batches { get; private set; }

    public bool Disposed { get; private set; }

    public IEnumerable<BsonDocument> Current { get; private set; } = [];

    public bool MoveNext(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (_position >= documents.Count)
        {
            Current = [];
            return false;
        }

        var take = Math.Min(driverBatchSize, documents.Count - _position);
        Current = documents.Skip(_position).Take(take).ToList();
        _position += take;
        Batches++;
        return true;
    }

    public Task<bool> MoveNextAsync(CancellationToken cancellationToken = default) => Task.FromResult(MoveNext(cancellationToken));

    public void Dispose() => Disposed = true;
}

public static class Documents
{
    public static List<BsonDocument> Daily(double[] values, string? key = null, double[]? x1 = null, int skipIndex = -1, bool numericTime = false)
    {
        var start = new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var list = new List<BsonDocument>();

        for (var i = 0; i < values.Length; i++)
        {
            if (i == skipIndex)
            {
                continue;
            }

            var document = new BsonDocument { ["value"] = values[i] };
            document["ts"] = numericTime ? (BsonValue)i : new BsonDateTime(start.AddDays(i));

            if (key is not null)
            {
                document["key"] = key;
            }

            if (x1 is not null)
            {
                document["x1"] = x1[i];
            }

            list.Add(document);
        }

        return list;
    }
}
