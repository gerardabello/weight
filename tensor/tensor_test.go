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

	tp1 := Tensor{}
	assert.Error(tp1.Allocate(0), "Allocate with zero size should panic")

	tp2 := Tensor{}
	assert.Error(tp2.Allocate(1, 3, 6, 0, 2), "Allocate with a zero size dimension should panic")

	tp3 := Tensor{}
	assert.Error(tp3.Allocate(), "Allocate with no dimensions should panic")

	tp4 := Tensor{}
	assert.Error(tp4.Allocate(-1), "Allocate with negative size should panic")

	tp5 := Tensor{}
	assert.Error(tp5.Allocate(1, 3, -2, -1, 2), "Allocate with a negative size dimension should panic")

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
	assert.EqualValues(6, t2.DimToFlat(0, 2), "Expected value")
	assert.EqualValues(4, t2.DimToFlat(1, 1), "Expected value")
	assert.EqualValues(3, t2.DimToFlat(0, 1), "Expected value")
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

	for i := 1; i < 4; i++ {
		for j := 0; j < 10; j++ {
			size := generateRandomSize(i)
			t1 := generateRandomTensorSize(size...)
			t2 := generateRandomTensorSize(size...)

			subvals := make([]float64, SizeLength(size))
			for k := 0; k < len(subvals); k++ {
				subvals[k] = t1.Values[k] - t2.Values[k]
			}

			err := t1.Substract(t2)

			assert.NoError(err, "Substracting should return no error")
			assert.EqualValues(subvals, t1.Values, "Using Substract() should return the difference of all values")
		}
	}

	{
		t1 := generateRandomTensorSize(3, 4)
		t2 := generateRandomTensorSize(3, 3)

		err := t1.Substract(t2)
		assert.Error(err, "Substracting tensors of different sizes should return an error")
	}

	{
		t1 := generateRandomTensorSize(3)
		t2 := generateRandomTensorSize(3, 3)

		err := t1.Substract(t2)
		assert.Error(err, "Substracting tensors of different dimensions should return an error")
	}

}

func TestAdd(t *testing.T) {
	assert := assert.New(t)

	//Test adding one tensor to another
	for i := 1; i < 4; i++ {
		for j := 0; j < 10; j++ {
			size := generateRandomSize(i)
			t1 := generateRandomTensorSize(size...)
			t2 := generateRandomTensorSize(size...)

			addvals := make([]float64, SizeLength(size))
			for k := 0; k < len(addvals); k++ {
				addvals[k] = t1.Values[k] + t2.Values[k]
			}

			err := t1.Add(t2)

			assert.NoError(err, "Adding should return no error")
			assert.EqualValues(addvals, t1.Values, "Using Add() should return the sum of all values")
		}
	}

	//Test adding multiple tensors to another
	for i := 1; i < 4; i++ {
		for j := 0; j < 20; j++ {
			size := generateRandomSize(i)
			t1 := generateRandomTensorSize(size...)
			t2 := generateRandomTensorSize(size...)
			t3 := generateRandomTensorSize(size...)
			t4 := generateRandomTensorSize(size...)

			addvals := make([]float64, SizeLength(size))
			for k := 0; k < len(addvals); k++ {
				addvals[k] = t1.Values[k] + t2.Values[k] + t3.Values[k] + t4.Values[k]
			}

			err := t1.Add(t2, t3, t4)

			assert.NoError(err, "Adding should return no error")
			assert.EqualValues(addvals, t1.Values, "Using Add() should return the sum of all values")
		}
	}

	{
		t1 := generateRandomTensorSize(3, 4)
		t2 := generateRandomTensorSize(3, 3)

		err := t1.Add(t2)
		assert.Error(err, "Adding tensors of different sizes should return an error")
	}

	{
		t1 := generateRandomTensorSize(3)
		t2 := generateRandomTensorSize(3, 3)

		err := t1.Add(t2)
		assert.Error(err, "Adding tensors of different dimensions should return an error")
	}

	{
		t1 := generateRandomTensorSize(5, 5)
		t2 := generateRandomTensorSize(5, 5)
		t3 := generateRandomTensorSize(5, 4)

		err := t1.Add(t2, t3)
		assert.Error(err, "Adding tensors of different sizes should return an error")
	}

	{
		t1 := generateRandomTensorSize(2, 2)
		t2 := generateRandomTensorSize(2, 2)
		t3 := generateRandomTensorSize(2, 2, 1)

		err := t1.Add(t2, t3)
		assert.Error(err, "Adding tensors of different dimensions should return an error")
	}
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

	for i := 1; i < 4; i++ {
		for j := 0; j < 10; j++ {
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

	{
		t1 := Tensor{
			Values: []float64{
				0, 0, 0,
				1, 0.5, 1,
				0, 1.1, 0},
			Size: []int{3, 3},
		}
		i, v := t1.Max()
		assert.EqualValues(7, i, "Max index incorrect")
		assert.EqualValues(1.1, v, "Max value incorrect")
	}

	{
		t1 := Tensor{
			Values: []float64{
				0, 0, 0,
				1, 0.5, -1,
				0, -1.1, 0},
			Size: []int{3, 3},
		}
		i, v := t1.Max()
		assert.EqualValues(3, i, "Max index incorrect")
		assert.EqualValues(1, v, "Max value incorrect")
	}
	{
		t1 := Tensor{
			Values: []float64{
				0, 0, 0,
				1, 0.5, -1,
				0, -1.1, 0,
				0, 0, -6,
				4, 0.5, -1,
				0, -1.1, 0,
			},
			Size: []int{3, 3, 2},
		}
		i, v := t1.Max()
		assert.EqualValues(12, i, "Max index incorrect")
		assert.EqualValues(4, v, "Max value incorrect")
	}

}

func TestMaxAbs(t *testing.T) {
	assert := assert.New(t)

	{
		t1 := Tensor{
			Values: []float64{
				0, 0, 0,
				1, 0.5, 1,
				0, 1.1, 0},
			Size: []int{3, 3},
		}
		i, v := t1.MaxAbs()
		assert.EqualValues(7, i, "Max index incorrect")
		assert.EqualValues(1.1, v, "Max value incorrect")
	}

	{
		t1 := Tensor{
			Values: []float64{
				0, 0, 0,
				1, 0.5, -1,
				0, -1.1, 0},
			Size: []int{3, 3},
		}
		i, v := t1.MaxAbs()
		assert.EqualValues(7, i, "Max index incorrect")
		assert.EqualValues(-1.1, v, "Max value incorrect")
	}
	{
		t1 := Tensor{
			Values: []float64{
				0, 0, 0,
				1, 0.5, -1,
				0, -1.1, 0,
				0, 0, -6,
				4, 0.5, -1,
				0, -1.1, 0,
			},
			Size: []int{3, 3, 2},
		}
		i, v := t1.MaxAbs()
		assert.EqualValues(11, i, "Max index incorrect")
		assert.EqualValues(-6, v, "Max value incorrect")
	}

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
