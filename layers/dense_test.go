package layers

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
	"time"

	"gitlab.com/gerardabello/weight/tensor"

	"github.com/stretchr/testify/assert"
)

func TestCorrectParamsSize(t *testing.T) {
	//As the size is random, do the test n times
	for i := 0; i < 10; i++ {
		inputSize := rand.Intn(1e3) + 1
		outputSize := rand.Intn(1e3) + 1

		fcl := NewDenseLayer([]int{inputSize}, []int{outputSize})

		if len(fcl.weights.Values) != outputSize*inputSize {
			t.Fatalf("Expected weights length to be %d (number of neurons), actual length : %d", outputSize*inputSize, len(fcl.weights.Values))
		}

		if len(fcl.bias.Values) != outputSize {
			t.Fatalf("Expected bias length to be %d (number of neurons), actual length : %d", outputSize, len(fcl.bias.Values))
		}
	}
}

func TestDenseRandomWeightsMean(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	//At least 2500 samples
	inputSize := rand.Intn(500) + 50
	outputSize := rand.Intn(500) + 50

	fcl := NewDenseLayer([]int{inputSize}, []int{outputSize})

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

//Test that the activation function checks that the input size is correct
func TestInputCheck(t *testing.T) {
	//test n times
	for i := 0; i < 50; i++ {
		inputSize := rand.Intn(50) + 50
		outputSize := rand.Intn(50) + 50
		dataSize := rand.Intn(50) + 50

		if inputSize == dataSize {
			//If they are equal it will not fail. We are not interested in this case
			continue
		}

		fcl := NewDenseLayer([]int{inputSize}, []int{outputSize})

		data := &tensor.Tensor{}
		data.Allocate(dataSize)

		_, err := fcl.Activate(data)

		if err == nil {
			t.Errorf("Activation with layer returned no error")
		}
	}
}

func TestDenseActivation(t *testing.T) {
	assert := assert.New(t)

	w := []float64{
		1.01, 3.1, 0.223,
		0.61, 2.902, 5.009}

	b := []float64{0.023, 0.82}

	layer := NewDenseLayer([]int{3}, []int{2})

	layer.weights.Values = w
	layer.bias.Values = b

	data := &tensor.Tensor{Size: []int{3}, Values: []float64{0.5, 0.123, 0.7784}}

	val, err := layer.Activate(data)

	if err != nil {
		t.Errorf("Error while activating layer: %s", err.Error())
	}

	assert.Equal(1, len(val.Size), "Dimension of output should be 1")
	assert.Equal(2, val.Size[0], "Output length should be 2")
	assert.Equal(val.Size[0], len(val.Values), "Output data length should be the same as declated in its size")

	assert.InDeltaSlice([]float64{
		0.023 + (0.5*1.01 + 0.123*3.1 + 0.7784*0.223),
		0.82 + (0.5*0.61 + 0.123*2.902 + 0.7784*5.009),
	}, val.Values, 1e-3, "Expected activation of neurons")
}

func TestDenseBackPropagation(t *testing.T) {
	assert := assert.New(t)

	w := []float64{
		-0.5910955, 0.75623487, -0.94522481, 0.34093502}

	b := []float64{0}

	layer := NewDenseLayer([]int{4}, []int{1})

	layer.weights.Values = w
	layer.bias.Values = b

	gradients := &tensor.Tensor{Size: []int{1}, Values: []float64{-0.11810546}}

	inputs := &tensor.Tensor{Size: []int{4}, Values: []float64{0.44856632, 0.51939863, 0.45968497, 0.59156505}}

	_, err := layer.Activate(inputs)
	if err != nil {
		t.Errorf("Error while backpropagatin error: %s", err.Error())
	}

	bp, err := layer.BackPropagate(gradients)
	if err != nil {
		t.Errorf("Error while backpropagatin error: %s", err.Error())
	}

	assert.InDeltaSlice([]float64{0.0698116, -0.08931546, 0.11163621, -0.04026629}, bp.Values, 1e-3, "Expected backpropagation")

}

func TestMarshalDense(t *testing.T) {
	assert := assert.New(t)

	fcl := NewDenseLayer([]int{3, 3, 6}, []int{5, 5})

	var b bytes.Buffer

	fcl.Marshal(&b)

	nfcl, err := UnmarshalDenseLayer(&b)

	assert.NoError(err, "Error while unmarshaling")

	assert.InDeltaSlice(nfcl.weights.Values, fcl.weights.Values, 1e-6, "Expected same weights")
	assert.InDeltaSlice(nfcl.weights.Size, fcl.weights.Size, 1e-6, "Expected same weights size")
	assert.InDeltaSlice(nfcl.bias.Values, fcl.bias.Values, 1e-6, "Expected same bias")
	assert.InDeltaSlice(nfcl.bias.Size, fcl.bias.Size, 1e-6, "Expected same bias size")

	assert.InDeltaSlice(nfcl.GetInputSize(), fcl.GetInputSize(), 1e-6, "Expected same input size")
	assert.InDeltaSlice(nfcl.GetOutputSize(), fcl.GetOutputSize(), 1e-6, "Expected same output size")

}
