package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSlice(t *testing.T) {
	assert := assert.New(t)

	tt := Tensor{
		Values: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
		Size:   []int{2, 3, 4},
	}

	assert.InDeltaSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, tt.Slice().Values, 0.001, "Expected equal")
	assert.InDeltaSlice([]int{2, 3, 4}, tt.Slice().Size, 0.001, "Expected equal")
	assert.InDeltaSlice([]float64{1, 2, 3, 4, 5, 6}, tt.Slice(0).Values, 0.001, "Expected equal")
	assert.InDeltaSlice([]int{2, 3}, tt.Slice(0).Size, 0.001, "Expected equal")
	assert.InDeltaSlice([]float64{19, 20, 21, 22, 23, 24}, tt.Slice(3).Values, 0.001, "Expected equal")
	assert.InDeltaSlice([]int{2, 3}, tt.Slice(3).Size, 0.001, "Expected equal")
	assert.InDeltaSlice([]float64{13, 14, 15, 16, 17, 18}, tt.Slice(2).Values, 0.001, "Expected equal")
	assert.InDeltaSlice([]int{2, 3}, tt.Slice(2).Size, 0.001, "Expected equal")
	assert.InDeltaSlice([]float64{13, 14}, tt.Slice(2).Slice(0).Values, 0.001, "Expected equal")
	assert.InDeltaSlice([]int{2}, tt.Slice(2).Slice(0).Size, 0.001, "Expected equal")
	assert.InDeltaSlice([]float64{3, 4}, tt.Slice(0).Slice(1).Values, 0.001, "Expected equal")
	assert.InDeltaSlice([]int{2}, tt.Slice(0).Slice(1).Size, 0.001, "Expected equal")
	assert.InDeltaSlice([]float64{3, 4}, tt.Slice(0, 1).Values, 0.001, "Expected equal")
	assert.InDeltaSlice([]int{2}, tt.Slice(0, 1).Size, 0.001, "Expected equal")
	assert.InDeltaSlice([]float64{17, 18}, tt.Slice(2, 2).Values, 0.001, "Expected equal")
	assert.InDeltaSlice([]int{2}, tt.Slice(2, 2).Size, 0.001, "Expected equal")
	assert.InDeltaSlice([]float64{17}, tt.Slice(2, 2, 0).Values, 0.001, "Expected equal")
	assert.InDeltaSlice([]int{}, tt.Slice(2, 2, 0).Size, 0.001, "Expected equal")
	assert.InDeltaSlice([]float64{18}, tt.Slice(2, 2, 1).Values, 0.001, "Expected equal")
	assert.InDeltaSlice([]int{}, tt.Slice(2, 2, 1).Size, 0.001, "Expected equal")

}
