package layers

import "gitlab.com/gerardabello/weight/tensor"
import "gitlab.com/gerardabello/weight"

type RecurrentLayer struct {
	Input, State, Output weight.BPLearnerLayer

	state *tensor.Tensor
	out   *tensor.Tensor
}

func NewRecurrentLayer(input, state, output weight.BPLearnerLayer) *RecurrentLayer {
	l := &RecurrentLayer{Input: input, State: state, Output: output}

	l.state = tensor.NewTensor(l.State.GetInputSize()...)

	l.out = tensor.NewTensor(l.Output.GetOutputSize()...)

	return l

}

func (l *RecurrentLayer) Activate(input *tensor.Tensor) (*tensor.Tensor, error) {

	var err error

	input, err = l.Input.Activate(input)

	l.state, err = l.State.Activate(l.state)
	l.state.Add(input)

	l.out, err = l.Output.Activate(input)

	return l.out, nil
}

func (l *RecurrentLayer) BackPropagate(err *tensor.Tensor) (*tensor.Tensor, error) {

	return &l.propagation, nil
}
