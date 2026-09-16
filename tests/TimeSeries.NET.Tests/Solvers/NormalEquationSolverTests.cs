using TimeSeries.Solvers;

namespace TimeSeries.Tests.Solvers;

public class NormalEquationSolverTests
{
    [Fact]
    public void TwoByTwo_HandComputed()
    {
        // G = [[4, 2], [2, 3]], x = [2, 1]  =>  b = [10, 7]
        double[] gram = [4, 2, 2, 3];
        double[] rhs = [10, 7];
        var x = new double[2];

        var ok = NormalEquationSolver.Default.TrySolve(gram, rhs, x, out var diagnostics);

        Assert.True(ok);
        Assert.True(diagnostics.Succeeded);
        Assert.Equal(-1, diagnostics.FailedColumn);
        Assert.Equal(2d, x[0], 1e-12);
        Assert.Equal(1d, x[1], 1e-12);
    }

    [Fact]
    public void StraightLineRegression_HandComputed()
    {
        // y = 1 + 2x at x = 1..4  =>  y = 3, 5, 7, 9
        // X'X = [[4, 10], [10, 30]],  X'y = [24, 70]
        double[] gram = [4, 10, 10, 30];
        double[] rhs = [24, 70];
        var beta = new double[2];

        Assert.True(NormalEquationSolver.Default.TrySolve(gram, rhs, beta, out _));
        Assert.Equal(1d, beta[0], 1e-12);
        Assert.Equal(2d, beta[1], 1e-12);
    }

    [Fact]
    public void ThreeRegressors_HandComputed()
    {
        // y = 1 + 2 x1 - 3 x2 at (x1, x2) = (0,0) (1,0) (0,1) (1,1) (2,1)  =>  y = 1, 3, -2, 0, 2
        // X'X = [[5,4,3],[4,6,3],[3,3,3]],  X'y = [4, 7, 0]
        double[] gram = [5, 4, 3, 4, 6, 3, 3, 3, 3];
        double[] rhs = [4, 7, 0];
        var beta = new double[3];

        Assert.True(NormalEquationSolver.Default.TrySolve(gram, rhs, beta, out _));
        Assert.Equal(1d, beta[0], 1e-12);
        Assert.Equal(2d, beta[1], 1e-12);
        Assert.Equal(-3d, beta[2], 1e-12);
    }

    [Fact]
    public void SingularSystem_ReturnsFalseNamingTheColumn()
    {
        // Second column is twice the first.
        double[] gram = [1, 2, 2, 4];
        double[] rhs = [3, 6];
        var x = new double[2];

        var ok = NormalEquationSolver.Default.TrySolve(gram, rhs, x, out var diagnostics);

        Assert.False(ok);
        Assert.False(diagnostics.Succeeded);
        Assert.Equal(1, diagnostics.FailedColumn);
        Assert.Equal(double.PositiveInfinity, diagnostics.PivotRatio);
    }

    [Fact]
    public void BadlyScaledColumns_SolveToRelativePrecision_WithAndWithoutScaling()
    {
        // Columns six orders of magnitude apart, x = [3e-6, 5]
        double[] gram = [1e12, 1e6, 1e6, 2];
        double[] rhs = [8e6, 13];

        var scaled = new double[2];
        var unscaled = new double[2];

        Assert.True(new NormalEquationSolver(scaleColumns: true).TrySolve(gram, rhs, scaled, out var withScaling));
        Assert.True(new NormalEquationSolver(scaleColumns: false).TrySolve(gram, rhs, unscaled, out var withoutScaling));

        Assert.Equal(3e-6, scaled[0], 3e-6 * 1e-9);
        Assert.Equal(5d, scaled[1], 5e-9);
        Assert.Equal(3e-6, unscaled[0], 3e-6 * 1e-9);
        Assert.Equal(5d, unscaled[1], 5e-9);

        // Scaling is what makes the pivot ratio a meaningful conditioning signal.
        Assert.True(withScaling.ColumnsScaled);
        Assert.False(withoutScaling.ColumnsScaled);
        Assert.True(withScaling.PivotRatio < withoutScaling.PivotRatio);
    }

    [Fact]
    public void Ridge_IsRelativeToEachColumnsScale_HandComputed()
    {
        // Diagonal G with unequal scales; a relative ridge of 0.5 shrinks both by the same
        // factor 1/1.5, which an absolute ridge would not.
        double[] gram = [4, 0, 0, 1];
        double[] rhs = [4, 1];
        var x = new double[2];

        Assert.True(new NormalEquationSolver(ridge: 0.5).TrySolve(gram, rhs, x, out var diagnostics));

        Assert.Equal(0.5, diagnostics.Ridge);
        Assert.Equal(1d / 1.5, x[0], 1e-12);
        Assert.Equal(1d / 1.5, x[1], 1e-12);
    }

    [Fact]
    public void ZeroColumn_FailsWithoutRidge_SucceedsWithIt()
    {
        double[] gram = [1, 0, 0, 0];
        double[] rhs = [1, 0];
        var x = new double[2];

        Assert.False(NormalEquationSolver.Default.TrySolve(gram, rhs, x, out var failed));
        Assert.Equal(1, failed.FailedColumn);

        Assert.True(new NormalEquationSolver(ridge: 1e-6).TrySolve(gram, rhs, x, out _));
        Assert.Equal(1d / (1d + 1e-6), x[0], 1e-12);
        Assert.Equal(0d, x[1], 1e-12);
    }

    [Fact]
    public void RowPivoting_HandlesAZeroLeadingPivot()
    {
        // Leading entry is zero; without pivoting elimination would divide by it.
        // [[0, 1], [1, 0]] x = [2, 3]  =>  x = [3, 2]
        double[] gram = [0, 1, 1, 0];
        double[] rhs = [2, 3];
        var x = new double[2];

        Assert.True(new NormalEquationSolver(scaleColumns: false).TrySolve(gram, rhs, x, out _));
        Assert.Equal(3d, x[0], 1e-12);
        Assert.Equal(2d, x[1], 1e-12);
    }

    [Fact]
    public void EmptySystem_SucceedsTrivially()
    {
        Assert.True(NormalEquationSolver.Default.TrySolve([], [], [], out var diagnostics));
        Assert.Equal(0, diagnostics.Dimension);
    }

    [Fact]
    public void InconsistentSizes_Throw()
    {
        double[] gram = [1, 2, 3];
        double[] rhs = [1, 2];
        var x = new double[2];

        Assert.Throws<ArgumentException>(() => NormalEquationSolver.Default.TrySolve(gram, rhs, x, out _));
    }

    [Fact]
    public void InvalidOptions_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new NormalEquationSolver(ridge: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NormalEquationSolver(pivotTolerance: 0));
    }
}
