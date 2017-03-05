package layers

import (
	"errors"
	"fmt"
	"math"

	"github.com/gerardabello/weight"
	"github.com/gerardabello/weight/tensor"
)

type PoolLayer struct {
	BaseLayer

	kernelSize []int

	lastMax [][]int //an array of length equal to output.GetNumberOfValues(). Each value has an array of coortinates to the input position that set the max
}

func NewPoolLayer(inputSize, kernelSize []int) *PoolLayer {
	if len(inputSize) == 0 {
		panic("Cannot create dimensionless layer")
	}

	if len(inputSize) != len(kernelSize) {
		panic("The kernel should have the same number of dimensions as the input")
	}

	for i := 0; i < len(inputSize); i++ {
		if inputSize[i]%kernelSize[i] != 0 {
			fmt.Println(inputSize)
			fmt.Println(kernelSize)
			panic(fmt.Sprintf("Input size in dim %d is not divisible by kernel", i))
		}
	}

	pl := &PoolLayer{kernelSize: kernelSize}

	//calculate output size
	outputSize := make([]int, len(inputSize))
	for i := 0; i < len(outputSize); i++ {
		//We assume the division has modulus 0 (it should be checked in the creation)
		outputSize[i] = inputSize[i] / kernelSize[i]
	}

	pl.BaseLayer.Init(inputSize, outputSize)

	pl.id = "Pool-" + pl.id

	//initialize lastMax
	pl.lastMax = make([][]int, pl.output.GetNumberOfValues())
	for i := 0; i < len(pl.lastMax); i++ {
		pl.lastMax[i] = make([]int, len(pl.GetInputSize()))
	}

	return pl
}

func (l *PoolLayer) CreateSlave() weight.Layer {
	nl := NewPoolLayer(l.GetInputSize(), l.kernelSize)
	nl.id = l.ID()

	return nl
}

//Recursion in the wild
//pooling divides the input matrix into an integer number of kernels (l.kernelSize), finds the maximum of every kernel and stores the maximums in l.output
func (l *PoolLayer) pooling(input *tensor.Tensor, dim int, pos []int, outpos []int) error {
	if input.Size[dim]%l.kernelSize[dim] != 0 {
		return fmt.Errorf("Kernel size in dim %d is not the same as input", dim)
	}

	//We use pos and outpos but we could just store pos and calculate pos/kernelSize, maybe an improvement
	p := make([]int, len(pos))
	op := make([]int, len(outpos))
	copy(op, outpos)
	copy(p, pos)

	n := input.Size[dim]
	for i := 0; i < n; i++ {
		p[dim] = i
		op[dim] = i / l.kernelSize[dim]

		if dim == input.GetDims()-1 {
			//If last dimension, calculate max and return
			curVal := l.output.GetVal(op...)
			inpVal := input.GetVal(p...)
			if inpVal > curVal {
				l.output.SetVal(inpVal, op...)

				//Store the index of the value that sets the max
				flatpos := l.output.DimToFlat(op...)

				copy(l.lastMax[flatpos], p)
			}
		} else {
			err := l.pooling(input, dim+1, p, op)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (l *PoolLayer) Activate(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	if !input.HasSize(l.GetInputSize()) {
		return nil, errors.New("Input has wrong size")
	}

	l.output.Zero(-math.MaxFloat64)

	pos := make([]int, input.GetDims())
	outpos := make([]int, input.GetDims())
	err := l.pooling(input, 0, pos, outpos)
	if err != nil {
		return nil, err
	}

	l.mutex.Unlock()
	return &l.output, nil
}

func (l *PoolLayer) BackPropagate(err *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	e := l.BaseLayer.BackPropagate(err)
	if e != nil {
		return nil, e
	}

	errs := err.Values

	for i := 0; i < len(l.propagation.Values); i++ {
		l.propagation.Values[i] = 0
	}

	for i := 0; i < len(errs); i++ {
		l.propagation.SetVal(errs[i], l.lastMax[i]...)
	}

	l.mutex.Unlock()
	return &l.propagation, nil
}

func (l *PoolLayer) GetParamGradPointers() ([]*float64, []*float64) {
	return []*float64{}, []*float64{}
}
