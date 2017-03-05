package costs

import (
	"fmt"
	"math"

	"github.com/gerardabello/weight"
	"github.com/gerardabello/weight/tensor"
)

type CrossEntropyCost struct {
	size     []int
	lastGrad []float64
}

func NewCrossEntropyCostFunction(size ...int) *CrossEntropyCost {
	s := CrossEntropyCost{}

	s.size = size

	return &s
}

func (c *CrossEntropyCost) CreateSlave() weight.BPCostFunc {
	return &CrossEntropyCost{size: c.size}
}

func (c *CrossEntropyCost) Cost(input *tensor.Tensor, target *tensor.Tensor) float64 {
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
		in := math.Max(input.Values[i], 1e-10)
		cost -= target.Values[i] * math.Log(in)
		c.lastGrad[i] = target.Values[i] / in
	}

	if math.IsNaN(cost) {
		fmt.Printf("input:%v\n", input.Values)
		panic("NaN!")
	}

	return cost
}

func (c *CrossEntropyCost) BackPropagate() *tensor.Tensor {

	if c.lastGrad == nil {
		panic("Cost function cannot propagate error because it has not been activated or it has not been configured to retain inputs")
	}

	n := len(c.lastGrad)

	grad := &tensor.Tensor{}
	grad.Allocate(c.size...)

	for i := 0; i < n; i++ {
		grad.Values[i] = -c.lastGrad[i]
	}

	return grad
}
