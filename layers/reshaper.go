package layers

import (
	"github.com/gerardabello/weight"
	"github.com/gerardabello/weight/tensor"
)

type ReshaperLayer struct {
	BaseLayer
}

func NewReshaperLayer(inputSize []int, outputSize []int) *ReshaperLayer {

	layer := &ReshaperLayer{}

	layer.BaseLayer.Init(inputSize, outputSize)

	if layer.output.GetNumberOfValues() != layer.propagation.GetNumberOfValues() {
		panic("Reshaper layer cannot create or delete values. Number of values should be the same.")
	}

	return layer
}

func (l *ReshaperLayer) CreateSlave() weight.Layer {

	nl := NewReshaperLayer(l.GetInputSize(), l.GetOutputSize())

	nl.id = l.ID()

	return nl
}

func (l *ReshaperLayer) Activate(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	l.output.Values = input.Values

	l.mutex.Unlock()
	return &l.output, nil
}

func (l *ReshaperLayer) BackPropagate(err *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	l.propagation.Values = err.Values

	l.mutex.Unlock()
	return &l.propagation, nil
}

/*
//Reshaper should not return debug values
func (l *ReshaperLayer) GetDebugInfo() []map[string]interface{} {
	return []map[string]interface{}{}
}
*/
