package layers

import (
	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

type ReLULayer struct {
	BaseLayer

	negativeSlope float64
}

func NewLeakyReLULayer(size ...int) *ReLULayer {
	layer := &ReLULayer{}
	layer.BaseLayer.Init(size, size)
	layer.id = "LeakyReLU-" + layer.id
	layer.negativeSlope = 0.01
	return layer
}

func NewReLULayer(size ...int) *ReLULayer {
	layer := &ReLULayer{}
	layer.BaseLayer.Init(size, size)
	layer.id = "ReLU-" + layer.id
	layer.negativeSlope = 0
	return layer
}

func (l *ReLULayer) CreateSlave() weight.Layer {
	nl := NewReLULayer(l.GetInputSize()...)
	nl.id = l.ID()

	return nl
}

func (l *ReLULayer) Activate(input *tensor.Tensor) (*tensor.Tensor, error) {
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
			l.output.Values[i] = l.negativeSlope * inputs[i]
		}
	}

	l.mutex.Unlock()
	return &l.output, nil
}

func (l *ReLULayer) BackPropagate(err *tensor.Tensor) (*tensor.Tensor, error) {
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
			l.propagation.Values[i] = l.negativeSlope * errs[i]
		}
	}
	l.mutex.Unlock()
	return &l.propagation, nil
}
