package costs

import (
	"math"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

type SquareMeanCost struct {
	size     []int
	lastGrad []float64
}

func NewSquareMeanCostFunction(size ...int) *SquareMeanCost {
	s := SquareMeanCost{}

	s.size = size

	return &s
}

func (c *SquareMeanCost) CreateSlave() weight.BPCostFunc {
	return &SquareMeanCost{size: c.size}
}

func (c *SquareMeanCost) Cost(input *tensor.Tensor, target *tensor.Tensor) float64 {
	if input.GetDims() != len(c.size) {
		panic("Cost function input has not the correct shape")
	}

	if input.GetDims() != target.GetDims() || input.GetNumberOfValues() != input.GetNumberOfValues() {
		panic("Cost function input has not the same shape as target")
	}

	n := input.GetNumberOfValues()

	c.lastGrad = make([]float64, n)

	cost := 0.0
	for i := 0; i < n; i++ {
		d := target.Values[i] - input.Values[i]
		c.lastGrad[i] = -d
		cost += 0.5*math.Pow(d, 2)
	}
	cost /= 2          //to make the derivative simpler
	cost /= float64(n) //to make mean


	return cost
}

func (c *SquareMeanCost) BackPropagate() *tensor.Tensor {

	if c.lastGrad == nil {
		panic("Cost function cannot propagate error because it has not been activated or it has not been configured to retain inputs")
	}

	n := len(c.lastGrad)

	grad := &tensor.Tensor{}
	grad.Allocate(c.size...)

	for i := 0; i < n; i++ {
		grad.Values[i] = -c.lastGrad[i] / float64(n)
	}

	return grad
}
