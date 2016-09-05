package tensor

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAllocate(t *testing.T) {
	assert := assert.New(t)

	assert.Equal(false, true, "Test not implemented")
}

func TestDimToFlat(t *testing.T) {
	assert := assert.New(t)

	assert.Equal(false, true, "Test not implemented")
}

func TestFlatToDim(t *testing.T) {
	assert := assert.New(t)

	tt := generateRandomTensor()

	p := 4

	assert.EqualValues(tt.DimToFlat(tt.FlatToDim(p)...), p, "Should be equal")
}

func TestSubstract(t *testing.T) {
	assert := assert.New(t)

	assert.Equal(false, true, "Test not implemented")
}

func TestZero(t *testing.T) {
	assert := assert.New(t)

	for i := 0; i < 10; i++ {
		t := generateRandomTensor()

		rv := (rand.Float64() - 0.5) * 2
		t.Zero(rv)

		for j := 0; j < SizeLength(t.Size); j++ {
			assert.True(t.Values[j] == rv, "After Zero(%.3f) all values should be equal to %.3f", rv, rv)
		}
	}
}

func TestCopy(t *testing.T) {
	assert := assert.New(t)

	for i := 0; i < 10; i++ {
		t := generateRandomTensor()
		tc := t.Copy()
		assert.True(tc.HasSize(t.Size), "Copy should have the same size as original")

		for j := 0; j < SizeLength(t.Size); j++ {
			t.Values[j] += 0.5
		}

		for j := 0; j < SizeLength(t.Size); j++ {
			assert.True(t.Values[j] != tc.Values[j], "After modifying the original tensor, the copy should have different values")
		}

	}

}

func TestNewTensor(t *testing.T) {
	assert := assert.New(t)

	assert.Equal(false, true, "Test not implemented")
}

func TestHasSize(t *testing.T) {
	assert := assert.New(t)

	for n := 1; n < 4; n++ { //Max 4 dimensions
		for i := 0; i < 10; i++ {
			s := generateRandomSize(n)
			assert.True(NewTensor(s...).HasSize(s), "HasSize should return to when passed the same size used in NewTensor (%v).", s)
		}
	}
}

func TestMax(t *testing.T) {
	assert := assert.New(t)

	assert.Equal(false, true, "Test not implemented")
}

type SizeSlice struct {
	slice []int
	size  int
}

func TestSizeLength(t *testing.T) {
	assert := assert.New(t)

	results := []SizeSlice{
		{[]int{1}, 1},
		{[]int{2}, 2},
		{[]int{100}, 100},
		{[]int{1, 2, 3}, 6},
		{[]int{1, 1, 1, 100, 1}, 100},
		{[]int{1, 3, 1, 100, 2}, 600},
		{[]int{100, 3, 100, 100, 2}, 6000000},
		{[]int{2, 3, 2, 10, 2}, 240},
	}

	for _, sl := range results {
		assert.EqualValues(sl.size, SizeLength(sl.slice), "SizeLength of %v should be %d", sl.slice, sl.size)
	}

	assert.Panics(func() {
		_ = SizeLength([]int{2, 3, 0})
	}, "Passing a slice with a 0 should panic")

	assert.Panics(func() {
		_ = SizeLength([]int{2, -3, 1})
	}, "Passing a slice with negative values should panic")

}

//Utils

func generateRandomTensorSize(size ...int) *Tensor {
	t := &Tensor{}
	t.Allocate(size...)

	for i := 0; i < t.GetNumberOfValues(); i++ {
		t.Values[i] = (rand.Float64() - 0.5) * 100
	}

	return t
}

func generateRandomTensorDims(dims int) *Tensor {
	return generateRandomTensorSize(generateRandomSize(dims)...)
}

func generateRandomSize(dims int) []int {
	size := make([]int, dims)

	for i := 0; i < dims; i++ {
		size[i] = rand.Intn(64) + 1
	}

	return size
}

func generateRandomTensor() *Tensor {
	return generateRandomTensorDims(rand.Intn(4) + 1)
}
