using TimeSeries.Accumulators;

namespace TimeSeries.Tests.Accumulators;

public class LagWindowTests
{
    [Fact]
    public void Row_HoldsTheCurrentObservationThenItsLags()
    {
        var window = new LagWindow(lagDepth: 3);
        var row = new double[4];

        foreach (var value in new double[] { 10, 20, 30, 40, 50 })
        {
            window.Push(value);
        }

        var count = window.CopyRow(row);

        Assert.Equal(4, count);
        Assert.Equal(new double[] { 50, 40, 30, 20 }, row);
    }

    [Fact]
    public void BeforeItFills_OnlyTheAvailableLagsAreReported()
    {
        var window = new LagWindow(lagDepth: 3);
        var row = new double[4];

        window.Push(10);
        Assert.Equal(1, window.Available);
        Assert.False(window.IsFull);
        Assert.Equal(1, window.CopyRow(row));
        Assert.Equal(10, row[0]);

        window.Push(20);
        Assert.Equal(2, window.CopyRow(row));
        Assert.Equal(new double[] { 20, 10 }, row.AsSpan(0, 2).ToArray());

        window.Push(30);
        window.Push(40);
        Assert.True(window.IsFull);
        Assert.Equal(4, window.Available);
    }

    [Fact]
    public void ItWrapsWithoutDrift_OverManyPushes()
    {
        var window = new LagWindow(lagDepth: 5);
        var row = new double[6];

        for (var i = 1; i <= 1000; i++)
        {
            window.Push(i);
        }

        window.CopyRow(row);

        Assert.Equal(new double[] { 1000, 999, 998, 997, 996, 995 }, row);
    }

    [Fact]
    public void LagDepthZero_YieldsTheObservationAlone()
    {
        var window = new LagWindow(lagDepth: 0);
        var row = new double[1];

        window.Push(7);
        window.Push(9);

        Assert.Equal(1, window.CopyRow(row));
        Assert.Equal(9, row[0]);
    }

    [Fact]
    public void CopyRow_BeforeAnyPush_Throws()
    {
        var window = new LagWindow(lagDepth: 2);
        var row = new double[3];

        var threw = false;
        try
        {
            window.CopyRow(row);
        }
        catch (InvalidOperationException)
        {
            threw = true;
        }

        Assert.True(threw);
    }

    [Fact]
    public void NegativeLagDepth_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LagWindow(-1));
    }

    [Fact]
    public void Reset_ClearsTheWindow()
    {
        var window = new LagWindow(lagDepth: 2);
        window.Push(1);
        window.Push(2);
        window.Push(3);
        Assert.True(window.IsFull);

        window.Reset();

        Assert.False(window.IsFull);
        Assert.Equal(0, window.Available);
    }
}
