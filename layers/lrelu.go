package layers

import (
	"archive/tar"
	"io"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

type LeakyReLULayer struct {
	BaseLayer
}

func NewLeakyReLULayer(size ...int) *LeakyReLULayer {
	layer := &LeakyReLULayer{}
	layer.BaseLayer.Init(size, size)
	layer.id = "LeakyReLU-" + layer.id
	return layer
}

func (l *LeakyReLULayer) CreateSlave() weight.Layer {
	nl := NewLeakyReLULayer(l.GetInputSize()...)
	nl.id = l.ID()

	return nl
}

func (l *LeakyReLULayer) Activate(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	err := l.BaseLayer.Activate(input)
	if err != nil {
		return nil, err
	}

	//New a slice with length equal to the number of neurons
	inputs := input.Values

	for i := range l.output.Values {
		if inputs[i] > 0 {
			l.output.Values[i] = inputs[i]
		} else {
			l.output.Values[i] = 0.01 * inputs[i]
		}
	}

	l.mutex.Unlock()
	return &l.output, nil
}

func (l *LeakyReLULayer) BackPropagate(err *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	e := l.BaseLayer.BackPropagate(err)
	if e != nil {
		return nil, e
	}

	errs := err.Values
	inputs := l.lastInput.Values

	for i := range l.propagation.Values {
		if inputs[i] > 0 {
			l.propagation.Values[i] = errs[i]
		} else {
			l.propagation.Values[i] = 0.01 * errs[i]
		}
	}
	l.mutex.Unlock()
	return &l.propagation, nil
}

func (l *LeakyReLULayer) Marshal(writer io.Writer) error {
	tarfile := tar.NewWriter(writer)
	defer tarfile.Close()

	//save info
	err := writeInfoTar(
		tarfile,
		&map[string]interface{}{
			"input": l.GetInputSize(),
		},
	)
	if err != nil {
		return err
	}

	return nil
}
