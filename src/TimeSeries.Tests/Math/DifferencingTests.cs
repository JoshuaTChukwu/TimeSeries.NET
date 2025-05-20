using Xunit;
using TimeSeries.Core.Math;
using TimeSeries.Tests.TestUtils;

namespace TimeSeries.Tests.Math
{
    public class DifferencingTests
    {
        [Fact]
        public void Difference_WithValidInput_ReturnsExpectedResults()
        {
            double[] expected = [3, 4, 4];  // from TrendingSeries: [13-10, 17-13, 21-17]

            double[] result = Differencing.Difference(TestData.TrendingSeries, lag: 1);

            Assert.Equal(expected.Length, result.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], result[i], precision: 5);
        }

        [Fact]
        public void Difference_ThrowsForInvalidLag()
        {
            Assert.Throws<ArgumentException>(() =>
                Differencing.Difference(TestData.TrendingSeries, lag: 0));
        }

        [Fact]
        public void Difference_ThrowsForNullInput()
        {
            Assert.Throws<ArgumentException>(() =>
                Differencing.Difference(null!, lag: 1));
        }

        [Fact]
        public void Difference_ThrowsForShortSeries()
        {
            Assert.Throws<ArgumentException>(() =>
                Differencing.Difference(TestData.ShortSeries, lag: 3));
        }

        [Fact]
        public void InverseDifference_RestoresOriginalSeries()
        {
            double[] original = TestData.TrendingSeries;
            double[] differenced = Differencing.Difference(original, lag: 1);

            double[] restored = Differencing.InverseDifference(differenced, original, lag: 1);

            Assert.Equal(original.Length, restored.Length);
            for (int i = 0; i < original.Length; i++)
                Assert.Equal(original[i], restored[i], precision: 5);
        }


        [Fact]
        public void InverseDifference_ThrowsForInvalidLag()
        {
            double[] differenced = Differencing.Difference(TestData.TrendingSeries, lag: 1);
            Assert.Throws<ArgumentException>(() =>
                Differencing.InverseDifference(differenced, TestData.TrendingSeries, lag: 0));
        }

        [Fact]
        public void InverseDifference_WithBadDifference_Throws()
        {
            double[] badDiff = new double[] { 3 };
            Assert.Throws<InvalidOperationException>(() =>
                Differencing.InverseDifference(
                badDiff,
                TestData.LinearSeries,
                lag: 1,
                startIndex: 0,
                validateAgainstOriginal: true)
            );
        }
        [Fact]
        public void InverseDifference_MissingLagValues_ThrowsArgumentException()
        {
            double[] original = [5, 8];
            double[] diff = [3];

            // Start index too close to end
            Assert.Throws<ArgumentException>(() =>
                Differencing.InverseDifference(
                    diff,
                    originalData: original,
                    lag: 2,
                    validateAgainstOriginal: true));
        }

        [Fact]
        public void InverseDifference_TooLongDifferencedArray_ThrowsArgumentException()
        {
            double[] original = [5, 8, 12];
            double[] oversizedDiff = [3, 4, 5]; // too long

            Assert.Throws<ArgumentException>(() =>
                Differencing.InverseDifference(
                    oversizedDiff,
                    originalData: original,
                    lag: 1,
                    validateAgainstOriginal: true));
        }
        [Fact]
        public void InverseDifference_RebuildsCumulativeGrowthSeries()
        {
            // Original input is a raw trend: { 10, 13, 17, 21 }
            double[] original = TestData.TrendingSeries;

            double[] diff = Differencing.Difference(original, lag: 1);  // [3, 7, 11]
            double[] cumulativeDiff = Differencing.Difference(original, lag: 1, isCumulative: true); // [4, 8]

            // Now reconstruct cumulative trend starting from original[0]
            double[] restored = Differencing.InverseDifference(
                diff,
                originalData: original,
                lag: 1,
                isCumulative: true,
                validateAgainstOriginal: false);

            // We expect the cumulative trend of the differenced data + base
            double[] expected = Differencing.InverseDifference(
                cumulativeDiff,
                originalData: original,
                lag: 1,
                isCumulative: false,
                validateAgainstOriginal: false);

            Assert.Equal(expected.Length, restored.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], restored[i], precision: 5);
        }

        [Fact]
        public void InverseDifference_TooShortOriginalData_Throws()
        {
            double[] diff = [1, 2];
            double[] original = [5]; // length < lag

            Assert.Throws<ArgumentException>(() =>
                Differencing.InverseDifference(diff, original, lag: 2));
        }

        [Fact]
        public void InverseDifference_DataLengthMismatch_Throws()
        {
            double[] diff = [1, 2, 3, 4]; // too long
            double[] original = [10, 11, 12]; // not enough room to restore

            Assert.Throws<ArgumentException>(() =>
                Differencing.InverseDifference(
                    diff,
                    original,
                    lag: 1,
                    validateAgainstOriginal: true));
        }


        [Fact]
        public void InverseDifference_NullInputs_ThrowArgumentNullException()
        {
            double[] diff = [1, 2];

            Assert.Throws<ArgumentNullException>(() =>
                Differencing.InverseDifference(null!, diff, lag: 1));

            Assert.Throws<ArgumentNullException>(() =>
                Differencing.InverseDifference(diff, null!, lag: 1));
        }



        [Fact]
        public void InverseDifference_WithStartIndex_WorksCorrectly()
        {
            double[] original = [5, 10, 15, 20, 25];
            double[] slice = [15, 20, 25]; // start from index 2
            double[] diff = Differencing.Difference(slice, lag: 1);  // [5, 5]

            double[] restored = Differencing.InverseDifference(diff, original, lag: 1, startIndex: 2);

            Assert.Equal(slice.Length, restored.Length);
            for (int i = 0; i < slice.Length; i++)
                Assert.Equal(slice[i], restored[i], precision: 5);
        }
    }
}
