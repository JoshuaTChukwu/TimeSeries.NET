using Xunit;
using TimeSeries.Core.Math;

namespace TimeSeries.Tests.Math
{
    public class DifferencingExtensionsTests
    {
        [Fact]
        public void Difference_OrderTwo_ReturnsCorrectSecondOrderDiff()
        {
            var input = new double[] { 2, 5, 9, 14 }; // 1st diff: [3, 4, 5]; 2nd diff: [1, 1]
            var expected = new double[] { 1, 1 };

            var result = input.Difference(order: 2);

            Assert.Equal(expected.Length, result.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], result[i], precision: 5);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        public void Difference_InvalidOrder_Throws(int order)
        {
            var input = new double[] { 1, 2, 3 };
            Assert.Throws<ArgumentException>(() => input.Difference(order));
        }

        [Fact]
        public void Difference_TooShortInput_Throws()
        {
            var input = new double[] { 3, 7 };
            Assert.Throws<ArgumentException>(() => input.Difference(order: 2));
        }
    }
}
