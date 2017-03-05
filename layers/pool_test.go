package layers

import (
	"testing"

	"github.com/gerardabello/weight/tensor"

	"github.com/stretchr/testify/assert"
)

func TestPoolActivation(t *testing.T) {

	assert := assert.New(t)

	layer := NewPoolLayer([]int{4, 4, 2}, []int{2, 2, 1})

	data := &tensor.Tensor{Size: []int{4, 4, 2},
		Values: []float64{
			0.1, 0.2, 0.3, 0.4,
			0.5, 0.6, 0.7, 0.8,
			0.9, 1.0, 1.1, 1.2,
			1.3, 1.4, 1.5, 1.6,

			3.1, 3.2, 3.3, 3.4,
			3.5, 3.6, 3.7, 3.8,
			3.9, 4.0, 4.1, 4.2,
			4.3, 4.4, 4.5, 4.6,
		}}

	out, err := layer.Activate(data)

	if err != nil {
		t.Errorf("Error while activating layer: %s", err.Error())
	}

	assert.InDeltaSlice([]float64{0.6, 0.8, 1.4, 1.6, 3.6, 3.8, 4.4, 4.6}, out.Values, 1e-3, "Expected activation of neurons")
}

func TestPoolActivationNegative(t *testing.T) {

	assert := assert.New(t)

	layer := NewPoolLayer([]int{4, 4, 2}, []int{2, 2, 1})

	data := &tensor.Tensor{Size: []int{4, 4, 2},
		Values: []float64{
			-0.1, -0.2, 0.3, 0.4,
			-0.5, -0.6, 0.7, 0.8,
			0.9, 1.0, 1.1, 1.2,
			1.3, 1.4, 1.5, 1.6,

			3.1, 3.2, 3.3, 3.4,
			3.5, 3.6, 3.7, 3.8,
			3.9, 4.0, 4.1, 4.2,
			4.3, 4.4, 4.5, 4.6,
		}}

	out, err := layer.Activate(data)

	if err != nil {
		t.Errorf("Error while activating layer: %s", err.Error())
	}

	assert.InDeltaSlice([]float64{-0.1, 0.8, 1.4, 1.6, 3.6, 3.8, 4.4, 4.6}, out.Values, 1e-3, "Expected activation of neurons")
}

func TestPoolPropagation(t *testing.T) {

	assert := assert.New(t)

	layer := NewPoolLayer([]int{4, 4, 2}, []int{2, 2, 1})

	data := &tensor.Tensor{
		Size: []int{4, 4, 2},
		Values: []float64{
			0.1, 0.2, 0.3, 0.4,
			0.5, 0.6, 0.7, 0.8,
			0.9, 1.0, 1.1, 1.2,
			1.3, 1.4, 1.5, 1.6,

			3.1, 3.2, 3.3, 3.4,
			3.5, 3.6, 3.7, 3.8,
			3.9, 4.0, 4.1, 4.2,
			4.3, 4.4, 4.5, 4.6,
		}}

	out, err := layer.Activate(data)

	if err != nil {
		t.Fatalf("Error while activating layer: %s", err.Error())
	}

	assert.InDeltaSlice([]float64{0.6, 0.8, 1.4, 1.6, 3.6, 3.8, 4.4, 4.6}, out.Values, 1e-3, "Expected activation of neurons")

	errGrad := &tensor.Tensor{
		Size:   []int{2, 2, 2},
		Values: []float64{1, 2, 3, 4, 5, 6, 7, 8},
	}

	bpout, err := layer.BackPropagate(errGrad)

	if err != nil {
		t.Fatalf("Error while backpropagating layer: %s", err.Error())
	}

	assert.InDeltaSlice(
		[]float64{
			0, 0, 0, 0,
			0, 1, 0, 2,
			0, 0, 0, 0,
			0, 3, 0, 4,

			0, 0, 0, 0,
			0, 5, 0, 6,
			0, 0, 0, 0,
			0, 7, 0, 8,
		},
		bpout.Values, 1e-3, "Expected backpropagation of neurons")

}
