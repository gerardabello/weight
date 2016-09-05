package tensor

import "testing"

func BenchmarkDimToFlat(b *testing.B) {
	t := generateRandomTensorDims(4)

	for n := 0; n < b.N; n++ {
		t.DimToFlat(1, 1, 1, 1)
	}
}
