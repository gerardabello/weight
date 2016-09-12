package tensor

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAllocate(t *testing.T) {
	assert := assert.New(t)

	t1 := Tensor{}
	t1.Allocate(3)
	assert.EqualValues(3, len(t1.Values), "Lenght of underlying array should be the same as allocated size")
	assert.EqualValues(1, len(t1.Size), "Lenght of underlying size array should be the same as allocated dimensions")

	t2 := Tensor{}
	t2.Allocate(3, 3)
	assert.EqualValues(9, len(t2.Values), "Lenght of underlying array should be the same as allocated size")
	assert.EqualValues(2, len(t2.Size), "Lenght of underlying size array should be the same as allocated dimensions")

	t5 := Tensor{}
	t5.Allocate(3, 3, 1, 2, 4)
	assert.EqualValues(72, len(t5.Values), "Lenght of underlying array should be the same as allocated size")
	assert.EqualValues(5, len(t5.Size), "Lenght of underlying size array should be the same as allocated dimensions")

	assert.Panics(func() {
		tp1 := Tensor{}
		tp1.Allocate(0)
	}, "Allocate with zero size should panic")

	assert.Panics(func() {
		tp2 := Tensor{}
		tp2.Allocate(1, 3, 6, 0, 2)
	}, "Allocate with a zero size dimension should panic")

	assert.Panics(func() {
		tp3 := Tensor{}
		tp3.Allocate()
	}, "Allocate with no dimensions should panic")

	assert.Panics(func() {
		tp4 := Tensor{}
		tp4.Allocate(-1)
	}, "Allocate with negative size should panic")

	assert.Panics(func() {
		tp5 := Tensor{}
		tp5.Allocate(1, 3, -2, -1, 2)
	}, "Allocate with a negative size dimension should panic")

}

func TestDimToFlat(t *testing.T) {
	assert := assert.New(t)
	t1 := NewTensor(5)
	assert.EqualValues(3, t1.DimToFlat(3), "Expected value")
	assert.EqualValues(0, t1.DimToFlat(0), "Expected value")
	assert.Panics(func() { t1.DimToFlat() }, "DimToFlat with no arguments should panic")
	assert.Panics(func() { t1.DimToFlat(1, 0) }, "DimToFlat with too many arguments should panic")
	assert.Panics(func() { t1.DimToFlat(-1) }, "DimToFlat with negative arguments should panic")
	assert.Panics(func() { t1.DimToFlat(10) }, "DimToFlat with out-of-bounds arguments should panic")

	t2 := NewTensor(3, 3)
	assert.EqualValues(0, t2.DimToFlat(0, 0), "Expected value")
	assert.EqualValues(6, t2.DimToFlat(2, 0), "Expected value")
	assert.EqualValues(4, t2.DimToFlat(1, 1), "Expected value")
	assert.EqualValues(1, t2.DimToFlat(0, 1), "Expected value")
	assert.Panics(func() { t2.DimToFlat() }, "DimToFlat with no arguments should panic")
	assert.Panics(func() { t2.DimToFlat(1, 0, 4) }, "DimToFlat with too many arguments should panic")
	assert.Panics(func() { t2.DimToFlat(1) }, "DimToFlat with too few arguments should panic")
	assert.Panics(func() { t2.DimToFlat(0, -1) }, "DimToFlat with negative arguments should panic")
	assert.Panics(func() { t2.DimToFlat(2, 9) }, "DimToFlat with out-of-bounds arguments should panic")

	t3 := NewTensor(5, 5, 5)
	assert.EqualValues(0, t3.DimToFlat(0, 0, 0), "Expected value")
	assert.EqualValues(3, t3.DimToFlat(3, 0, 0), "Expected value")
	assert.EqualValues(8, t3.DimToFlat(3, 1, 0), "Expected value")
	assert.EqualValues(58, t3.DimToFlat(3, 1, 2), "Expected value")
	assert.EqualValues(55, t3.DimToFlat(0, 1, 2), "Expected value")
	assert.Panics(func() { t3.DimToFlat() }, "DimToFlat with no arguments should panic")
	assert.Panics(func() { t3.DimToFlat(1, 0, 4, 5) }, "DimToFlat with too many arguments should panic")
	assert.Panics(func() { t3.DimToFlat(1, 0) }, "DimToFlat with too few arguments should panic")
	assert.Panics(func() { t3.DimToFlat(0, 0, -1) }, "DimToFlat with negative arguments should panic")
	assert.Panics(func() { t3.DimToFlat(2, 3, 9) }, "DimToFlat with out-of-bounds arguments should panic")
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
	//As this just calls Allocate, we just test that this returns tensors equal to Allocate
	assert := assert.New(t)

	for i := 0; i < 20; i++ {
		for j := 0; j < 20; j++ {
			size := generateRandomSize(i)
			t1 := NewTensor(size...)
			t2 := Tensor{}
			t2.Allocate(size...)
			assert.EqualValues(len(t2.Values), len(t1.Values), "Lenght of underlying array should be the same as with Allocate()")
			assert.EqualValues(len(t2.Size), len(t1.Size), "Lenght of underlying size array should be the same as with Allocate()")
		}
	}
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
