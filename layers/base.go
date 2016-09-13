package layers

import (
	"errors"
	"fmt"
	"sync"

	"gitlab.com/gerardabello/weight/debug"
	"gitlab.com/gerardabello/weight/tensor"
)

type BaseLayer struct {
	id string

	output      tensor.Tensor
	propagation tensor.Tensor

	//Not all layers need those, but they are quite common and we can provide common functionality. Layers like SigmoidLayer that do not need them just leave them at nil.
	weights     *tensor.Tensor
	bias        *tensor.Tensor
	weightsGrad *tensor.Tensor
	biasGrad    *tensor.Tensor

	lastInput *tensor.Tensor

	mutex *sync.Mutex
}

func (l *BaseLayer) Init(inputSize, outputSize []int) error {
	l.mutex = &sync.Mutex{}

	l.id = RandomID(8)

	if len(inputSize) <= 0 || len(outputSize) <= 0 {
		return errors.New("Cannot create dimensionless layer")
	}

	var err error
	err = l.output.Allocate(outputSize...)
	if err != nil {
		return err
	}
	err = l.propagation.Allocate(inputSize...)
	if err != nil {
		return err
	}

	return nil
}

func (l *BaseLayer) ID() string {
	return l.id
}

func (l *BaseLayer) GetOutputSize() []int {
	return l.output.Size
}

func (l *BaseLayer) GetInputSize() []int {
	return l.propagation.Size
}

func (l *BaseLayer) Activate(input *tensor.Tensor) error {
	if !input.HasSize(l.GetInputSize()) {
		return fmt.Errorf("Layer has input size %d but input is size %d", l.GetInputSize(), input.Size)
	}

	l.lastInput = input

	l.output.Zero(0)

	return nil
}

func (l *BaseLayer) BackPropagate(err *tensor.Tensor) error {
	l.propagation.Zero(0)

	return nil
}

func (l *BaseLayer) GetParamGradPointers() ([]*float64, []*float64) {
	params := []*float64{}
	grads := []*float64{}
	if l.bias != nil {

		for i := range l.bias.Values {
			params = append(params, &l.bias.Values[i])
			grads = append(grads, &l.biasGrad.Values[i])
		}
	}

	if l.weights != nil {

		for i := range l.weights.Values {
			params = append(params, &l.weights.Values[i])
			grads = append(grads, &l.weightsGrad.Values[i])
		}
	}
	return params, grads
}

func (l *BaseLayer) GetDebugInfo() []*debug.LayerInfo {
	ret := debug.LayerInfo{}
	l.mutex.Lock()

	ret.ID = l.ID()

	if l.weights != nil {
		ret.WeightsStats = l.weights.Stats()
		wim, err := l.weights.ImageSlice()
		if err == nil {
			ret.WeightsImg = wim
		}
	}

	if l.bias != nil {
		ret.BiasStats = l.bias.Stats()

		bim, err := l.bias.ImageSlice()
		if err == nil {
			ret.BiasImg = bim
		}
	}

	ret.OutStats = l.output.Stats()

	oim, err := l.output.ImageSlice()
	if err == nil {
		ret.OutImg = oim
	}

	ret.ErrStats = l.propagation.Stats()
	pim, err := l.propagation.ImageSlice()
	if err == nil {
		ret.ErrImg = pim
	}

	l.mutex.Unlock()

	return []*debug.LayerInfo{&ret}
}
