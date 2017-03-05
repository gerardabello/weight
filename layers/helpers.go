package layers

import "github.com/gerardabello/weight"

//News a block of Convolutionals, ReLUs and Pool
func NewCRPBlock(inputSize []int, nConv, nKernels int) *FFNet {

	if len(inputSize) == 3 {
		//ok
	} else {
		panic("Input size should have 3 dimensions")
	}

	lyrs := []weight.Layer{}

	size := inputSize
	for i := 0; i < nConv; i++ {
		//cl0 := NewSquareConvolutionalLayer(28, 1, 16, 2, 1, 2)
		cl := NewConvolutionalLayer(size[0], size[1], size[2],
			nKernels,
			1, 1,
			1, 1,
			1, 1)

		lyrs = append(lyrs,
			cl,
			NewReLULayer(cl.GetOutputSize()...),
		)

		size[2] = nKernels

	}

	if size[0]%2 != 0 || size[1]%2 != 0 {
		panic("Input size in CPRBlock should be divisible by 2")
	}

	lyrs = append(lyrs,
		NewPoolLayer(size, []int{2, 2, 1}),
	)

	net, err := NewSequentialNet(
		lyrs...,
	)

	if err != nil {
		panic(err)
	}

	return net

}

func NewCRPBlocks(inputSize []int, nConv, nKernels, nBlocks int) *FFNet {

	lyrs := make([]weight.Layer, nBlocks)

	lyrs[0] = NewCRPBlock(inputSize, nConv, nKernels)
	for i := 1; i < nBlocks; i++ {
		lyrs[i] = NewCRPBlock(lyrs[i-1].GetOutputSize(), nConv, nKernels)
	}

	net, err := NewSequentialNet(
		lyrs...,
	)

	if err != nil {
		panic(err)
	}

	return net
}
