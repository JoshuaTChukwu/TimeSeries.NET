using System.Data.Common;
using System.Globalization;
using Microsoft.Data.Sqlite;

namespace TimeSeries.Data.Tests;

/// <summary>
/// A private in-memory SQLite database that lives as long as this object: the shared
/// cache keeps it alive while the master connection is open, and every connection the
/// factory hands out reaches the same data.
/// </summary>
public sealed class SqliteDatabase : IDisposable
{
    private readonly SqliteConnection _master;

    public SqliteDatabase()
    {
        ConnectionString = $"Data Source=file:tsnet_{Guid.NewGuid():N}?mode=memory&cache=shared";
        _master = new SqliteConnection(ConnectionString);
        _master.Open();
    }

    public string ConnectionString { get; }

    /// <summary>What a consumer passes to <see cref="DbTimeSeriesSource"/>.</summary>
    public DbConnection Connect() => new SqliteConnection(ConnectionString);

    /// <summary>Tracks every connection handed out so tests can check they were disposed.</summary>
    public List<SqliteConnection> Issued { get; } = [];

    public DbConnection ConnectTracked()
    {
        var connection = new SqliteConnection(ConnectionString);
        Issued.Add(connection);
        return connection;
    }

    public void Execute(string sql)
    {
        using var command = _master.CreateCommand();
        command.CommandText = sql;
        command.ExecuteNonQuery();
    }

    /// <summary>
    /// Creates a table (id, ts, key, value, x1, x2) and fills it. Timestamps are ISO text
    /// at daily spacing from 2020-01-01, unless a row is given a numeric time instead.
    /// </summary>
    public void CreateSeriesTable(string table, IEnumerable<Row> rows)
    {
        Execute($"CREATE TABLE {table} (id INTEGER PRIMARY KEY, ts TEXT, t INTEGER, key TEXT, value REAL, x1 REAL, x2 REAL)");

        using var transaction = _master.BeginTransaction();
        using var command = _master.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = $"INSERT INTO {table} (ts, t, key, value, x1, x2) VALUES (@ts, @t, @key, @value, @x1, @x2)";

        var ts = command.Parameters.Add("@ts", SqliteType.Text);
        var t = command.Parameters.Add("@t", SqliteType.Integer);
        var key = command.Parameters.Add("@key", SqliteType.Text);
        var value = command.Parameters.Add("@value", SqliteType.Real);
        var x1 = command.Parameters.Add("@x1", SqliteType.Real);
        var x2 = command.Parameters.Add("@x2", SqliteType.Real);

        foreach (var row in rows)
        {
            ts.Value = row.Time is DateTime time ? time.ToString("O", CultureInfo.InvariantCulture) : DBNull.Value;
            t.Value = row.NumericTime is long numeric ? numeric : DBNull.Value;
            key.Value = row.Key is null ? DBNull.Value : row.Key;
            value.Value = row.Value is double v ? v : DBNull.Value;
            x1.Value = row.X1 is double a ? a : DBNull.Value;
            x2.Value = row.X2 is double b ? b : DBNull.Value;
            command.ExecuteNonQuery();
        }

        transaction.Commit();
    }

    public static IEnumerable<Row> Daily(double[] values, string? key = null, double[]? x1 = null, double[]? x2 = null, int skipIndex = -1)
    {
        var start = new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        for (var i = 0; i < values.Length; i++)
        {
            if (i == skipIndex)
            {
                continue;
            }

            yield return new Row(start.AddDays(i), i, key, values[i], x1?[i], x2?[i]);
        }
    }

    public void Dispose() => _master.Dispose();

    public sealed record Row(DateTime? Time, long? NumericTime, string? Key, double? Value, double? X1, double? X2);
}
