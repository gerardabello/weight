package layers

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/gerardabello/weight"
	"github.com/gerardabello/weight/tensor"
)

func TestConvRandomWeightsMean(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	//At least 2500 samples
	inputSize := rand.Intn(500) + 50
	outputSize := rand.Intn(500) + 50

	fcl := NewSquareConvolutionalLayer(inputSize, outputSize, outputSize, 1, 1, 0)

	var acc float64
	n := 0
	for _, w := range fcl.weights.Values {
		acc += w
		n++
	}

	mean := acc / float64(n)

	if math.Abs(float64(mean)) > 0.001 { //this number looks good
		t.Errorf("Mean of weights is too high: %f", mean)
	}
}

func TestConvolutionalActivation(t *testing.T) {
	{
		layer := NewSquareConvolutionalLayer(4, 1, 1, 1, 1, 0)

		layer.weights = &tensor.Tensor{
			Size: []int{3, 3, 1, 1},
			Values: []float64{
				0, 0, 0,
				1, 1, 0,
				0, 0, 0,
			},
		}
		data := &tensor.Tensor{
			Size: []int{4, 4, 1},
			Values: []float64{
				0.1, 0.2, 0.3, 0.4,
				0.5, 0.6, 0.7, 0.8,
				0.9, 1.0, 1.1, 1.2,
				1.3, 1.4, 1.5, 1.6,
			},
		}
		testConvolutionalActivation(t, layer, data, []float64{1.1, 1.3, 1.9, 2.1})
	}

	{
		layer := NewSquareConvolutionalLayer(4, 2, 1, 1, 1, 0)

		layer.weights = &tensor.Tensor{
			Size: []int{3, 3, 2, 1},
			Values: []float64{
				0, 0, 0,
				1, 1, 0,
				0, 0, 0,

				0, 0, 1,
				0, 0, 0,
				0, 0, 1,
			},
		}
		data := &tensor.Tensor{
			Size: []int{4, 4, 2},
			Values: []float64{
				0.1, 0.2, 0.3, 0.4,
				0.5, 0.6, 0.7, 0.8,
				0.9, 1.0, 1.1, 1.2,
				1.3, 1.4, 1.5, 1.6,

				2.1, 2.2, 2.3, 2.4,
				2.5, 2.6, 2.7, 2.8,
				2.9, 3.0, 3.1, 3.2,
				3.3, 3.4, 3.5, 3.6,
			},
		}
		testConvolutionalActivation(t, layer, data, []float64{6.5, 6.9, 8.1, 8.5})
	}

	{
		layer := NewSquareConvolutionalLayer(4, 2, 1, 1, 1, 0)

		layer.weights = &tensor.Tensor{
			Size: []int{3, 3, 2, 1},
			Values: []float64{
				0, 0, 0,
				1, 1, 0,
				0, 0, 0,

				0, 0, 1,
				0, 0, 0,
				0, 0, 1,
			},
		}
		layer.bias = &tensor.Tensor{
			Size: []int{1},
			Values: []float64{
				1,
			},
		}
		data := &tensor.Tensor{
			Size: []int{4, 4, 2},
			Values: []float64{
				0.1, 0.2, 0.3, 0.4,
				0.5, 0.6, 0.7, 0.8,
				0.9, 1.0, 1.1, 1.2,
				1.3, 1.4, 1.5, 1.6,

				2.1, 2.2, 2.3, 2.4,
				2.5, 2.6, 2.7, 2.8,
				2.9, 3.0, 3.1, 3.2,
				3.3, 3.4, 3.5, 3.6,
			},
		}
		testConvolutionalActivation(t, layer, data, []float64{7.5, 7.9, 9.1, 9.5})
	}

	{
		layer := NewSquareConvolutionalLayer(4, 2, 2, 1, 1, 0)

		layer.weights = &tensor.Tensor{
			Size: []int{3, 3, 2, 2},
			Values: []float64{
				0, 0, 0,
				1, 1, 0,
				0, 0, 0,

				0, 0, 1,
				0, 0, 0,
				0, 0, 1,

				0, 0, 0,
				1, 1, 0,
				0, 0, 0,

				0, 0, 1,
				0, 0, 0,
				0, 0, 1,
			},
		}

		layer.bias = &tensor.Tensor{
			Size: []int{2},
			Values: []float64{
				-0.5, 1,
			},
		}

		data := &tensor.Tensor{
			Size: []int{4, 4, 2},
			Values: []float64{
				0.1, 0.2, 0.3, 0.4,
				0.5, 0.6, 0.7, 0.8,
				0.9, 1.0, 1.1, 1.2,
				1.3, 1.4, 1.5, 1.6,

				2.1, 2.2, 2.3, 2.4,
				2.5, 2.6, 2.7, 2.8,
				2.9, 3.0, 3.1, 3.2,
				3.3, 3.4, 3.5, 3.6,
			},
		}
		testConvolutionalActivation(t, layer, data, []float64{6.0, 6.4, 7.6, 8.0, 7.5, 7.9, 9.1, 9.5})
	}
}

func testConvolutionalActivation(t *testing.T, layer weight.Layer, data *tensor.Tensor, expectedValues []float64) {
	assert := assert.New(t)

	val, err := layer.Activate(data)

	if err != nil {
		t.Errorf("Error while activating layer: %s", err.Error())
	}

	assert.InDeltaSlice(expectedValues, val.Values, 0.0001, "Conv activation not correct. output :%v expected:%v", val.Values, expectedValues)
}

func TestConvolutionalBackPropagation(t *testing.T) {

	layer := NewSquareConvolutionalLayer(3, 1, 1, 1, 1, 0)

	layer.weights = &tensor.Tensor{
		Size: []int{3, 3, 1, 1},
		Values: []float64{
			0, 0, 0,
			0.5, 1, 0,
			0, 0, 0,
		},
	}

	data := &tensor.Tensor{
		Size: []int{3, 3, 1}, Values: []float64{
			0.1, 0.2, 0.3,
			0.5, 0.6, 0.7,
			0.9, 1.0, 1.1,
		}}

	gradErr := &tensor.Tensor{
		Size: []int{1, 1, 1}, Values: []float64{
			0.4,
		}}

	_, err := layer.Activate(data)

	if err != nil {
		t.Fatalf("Error while activating layer: %s", err.Error())
	}

	grad, err := layer.BackPropagate(gradErr)

	if err != nil {
		t.Fatalf("Error while activating layer: %s", err.Error())
	}

	expectedValues := []float64{
		0, 0, 0,
		0.2, 0.4, 0,
		0, 0, 0,
	}

	assert.InDeltaSlice(t, expectedValues, grad.Values, 0.0001, "Conv bp not correct. output :%v expected:%v", grad.Values, expectedValues)

}

func TestIm2Col(t *testing.T) {

	assert := assert.New(t)

	data := &tensor.Tensor{
		Size: []int{4, 4, 1},
		Values: []float64{
			0.1, 0.2, 0.3, 0.4,
			0.5, 0.6, 0.7, 0.8,
			0.9, 1.0, 1.1, 1.2,
			1.3, 1.4, 1.5, 1.6,
		},
	}

	sx, sy := im2colSize(data.Size[0], data.Size[1], data.Size[2], 3, 3, 0, 0, 1, 1)

	out := make([]float64, sx*sy)

	im2col(data.Values, out, data.Size[0], data.Size[1], data.Size[2], 3, 3, 0, 0, 1, 1)

	assert.InDeltaSlice([]float64{0.1, 0.2, 0.5, 0.6, 0.2, 0.3, 0.6, 0.7, 0.3, 0.4, 0.7, 0.8, 0.5, 0.6, 0.9, 1, 0.6, 0.7, 1, 1.1, 0.7, 0.8, 1.1, 1.2, 0.9, 1, 1.3, 1.4, 1, 1.1, 1.4, 1.5, 1.1, 1.2, 1.5, 1.6}, out, 0.001, "Unexpected value afeter im2col")

}
