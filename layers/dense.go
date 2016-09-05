package layers

import (
	"archive/tar"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/rand"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/marshaling"
	"gitlab.com/gerardabello/weight/tensor"
)

//DenseLayer computes each output using a weighted sum of all inputs plus a bias
type DenseLayer struct {
	BaseLayer
}

//NewDenseLayer creates a new DenseLayer
func NewDenseLayer(inputSize, outputSize []int) *DenseLayer {

	layer := &DenseLayer{}

	layer.BaseLayer.Init(inputSize, outputSize)

	layer.id = "Dense-" + layer.id

	//allocate weights
	layer.weights = &tensor.Tensor{}
	layer.weightsGrad = &tensor.Tensor{}
	layer.weights.Allocate(layer.GetNumberOfInputs(), layer.GetNumberOfNeurons())
	layer.weightsGrad.Allocate(layer.GetNumberOfInputs(), layer.GetNumberOfNeurons())

	//initialize for relus
	stdev := math.Sqrt(2.0 / float64(tensor.SizeLength(layer.GetInputSize())))
	for i := range layer.weights.Values {
		//Initialize weights with uniform random from -variance to variance
		layer.weights.Values[i] = rand.NormFloat64() * stdev
	}

	//Initialize slice of biases, one for each neuron
	layer.bias = &tensor.Tensor{}
	layer.biasGrad = &tensor.Tensor{}
	layer.bias.Allocate(layer.GetNumberOfNeurons())
	layer.biasGrad.Allocate(layer.GetNumberOfNeurons())

	for i := range layer.bias.Values {
		//Leave bias to 0
		layer.bias.Values[i] = 0.0
	}

	return layer
}

//CreateSlave creates a slave of the DenseLayer. See EnslaverLayer in package weight for more information on layer slaves.
func (l *DenseLayer) CreateSlave() weight.Layer {
	nl := NewDenseLayer(l.GetInputSize(), l.GetOutputSize())

	//This slices are copied by value but the internal data is a pointer. As we will not change the lengths of those slices we can think of them like a pointer, so all copies will have the same input size and the same parameters.
	nl.weights = l.weights
	nl.bias = l.bias

	nl.id = l.ID()

	return nl
}

func (l *DenseLayer) GetNumberOfNeurons() int {
	return tensor.SizeLength(l.GetOutputSize())
}

func (l *DenseLayer) GetNumberOfInputs() int {
	return tensor.SizeLength(l.GetInputSize())
}

//Activate takes and input tensor and computes an output tensor where each value is the weighed sum of all the input values.
func (l *DenseLayer) Activate(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	err := l.BaseLayer.Activate(input)
	if err != nil {
		return nil, err
	}

	inputs := input.Values
	activation := l.output.Values

	na := len(activation)
	ni := l.GetNumberOfInputs()

	//Calculate forward pass
	for i := 0; i < na; i++ {
		w := l.weights.Values[ni*i : ni*(i+1)]
		a := l.bias.Values[i]
		for j, inp := range inputs {
			a = a + inp*w[j]
		}
		activation[i] = a
	}

	l.mutex.Unlock()
	return &l.output, nil
}

func (l *DenseLayer) BackPropagate(err *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	e := l.BaseLayer.BackPropagate(err)
	if e != nil {
		return nil, e
	}

	ni := l.GetNumberOfInputs()
	no := l.GetNumberOfNeurons()

	errs := err.Values
	inputs := l.lastInput.Values

	l.propagation.Zero(0)

	for i := 0; i < no; i++ {
		//Update bias
		e := errs[i]
		l.biasGrad.Values[i] += e

		//slices to make the inner loop faster
		w := l.weights.Values[ni*i : ni*(i+1)]
		wg := l.weightsGrad.Values[ni*i : ni*(i+1)]

		for j := 0; j < ni; j++ {
			//Update weights
			wg[j] += e * inputs[j]
			//Calculate backpropagation
			l.propagation.Values[j] += w[j] * e
		}
	}

	l.mutex.Unlock()
	return &l.propagation, nil
}

func (l *DenseLayer) Marshal(writer io.Writer) error {
	tarfile := tar.NewWriter(writer)
	defer tarfile.Close()

	//save info
	err := writeInfoTar(
		tarfile,
		&map[string]interface{}{
			"input":  l.GetInputSize(),
			"output": l.GetOutputSize(),
		},
	)
	if err != nil {
		return err
	}

	err = writeTensorTar(tarfile, l.bias, "bias")
	if err != nil {
		return err
	}

	err = writeTensorTar(tarfile, l.weights, "weights")
	if err != nil {
		return err
	}

	return nil

}

const DenseName = "dense"

func init() {
	marshaling.RegisterFormat(DenseName, UnmarshalDenseLayer)
}

func (l *DenseLayer) GetName() string {
	return DenseName
}

func UnmarshalDenseLayer(reader io.Reader) (weight.MarshalLayer, error) {
	tr := tar.NewReader(reader)
	// Iterate through the files in the archive.

	var inputSize []int
	var outputSize []int

	var weights *tensor.Tensor
	var bias *tensor.Tensor

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			// end of tar archive
			break
		}
		if err != nil {
			return nil, err
		}

		switch hdr.Name {
		case "info":
			info := map[string][]int{}
			dec := json.NewDecoder(tr)
			err = dec.Decode(&info)
			if err == io.EOF {
				continue
			}
			if err != nil {
				return nil, err
			}

			inputSize = info["input"]
			outputSize = info["output"]

		case "weights":
			weights, err = tensor.Unmarshal(tr)
			if err != nil {
				return nil, err
			}
		case "bias":
			bias, err = tensor.Unmarshal(tr)
			if err != nil {
				return nil, err

			}
		default:
			return nil, errors.New("Unrecognized file " + hdr.Name)
		}

	}

	l := NewDenseLayer(inputSize, outputSize)

	l.weights = weights
	l.bias = bias

	return l, nil
}
