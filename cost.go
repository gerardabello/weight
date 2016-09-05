package weight

import "gitlab.com/gerardabello/weight/tensor"

//CostFunc is an object capable of computing the loss value given a result and the correct answer
type CostFunc interface {
	Cost(*tensor.Tensor, *tensor.Tensor) float64
}

//BPCostFunc is a CostFunc than can backpropagate error
type BPCostFunc interface {
	CostFunc
	CreateSlave() BPCostFunc
	BackPropagate() *tensor.Tensor
}
