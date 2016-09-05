package layers

import (
	"math"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

type SigmoidLayer struct {
	BaseLayer
}

func NewSigmoidLayer(size ...int) *SigmoidLayer {
	layer := &SigmoidLayer{}
	layer.BaseLayer.Init(size, size)
	return layer
}

func (l *SigmoidLayer) CreateSlave() weight.Layer {
	nl := NewSigmoidLayer(l.GetInputSize()...)
	nl.id = l.ID()

	return nl
}

func (l *SigmoidLayer) Activate(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	err := l.BaseLayer.Activate(input)
	if err != nil {
		return nil, err
	}

	inputs := input.Values

	for i := range l.output.Values {
		l.output.Values[i] = 1 / (1 + math.Exp(-inputs[i]))
	}

	l.mutex.Unlock()
	return &l.output, nil
}

func (l *SigmoidLayer) BackPropagate(err *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	e := l.BaseLayer.BackPropagate(err)
	if e != nil {
		return nil, e
	}

	errs := err.Values
	inputs := l.lastInput.Values

	for i := range l.propagation.Values {
		l.propagation.Values[i] = math.Exp(-inputs[i]) / math.Pow(1+math.Exp(-inputs[i]), 2) * errs[i]
	}

	l.mutex.Unlock()
	return &l.propagation, nil
}
