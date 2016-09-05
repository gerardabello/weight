package layers

import (
	"math/rand"
	"runtime"
	"testing"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

//https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
//Forward time per example: ~8ms
//Backprop time per example: ~11ms
func createCIFAR10ConvNetJSDemoNet() (*FFNet, error) {
	cl0, err := NewSquareConvolutionalLayer(32, 3, 16, 2, 1, 2)
	if err != nil {
		return nil, err
	}

	cl1, err := NewSquareConvolutionalLayer(16, 16, 20, 2, 1, 2)
	if err != nil {
		return nil, err
	}

	cl2, err := NewSquareConvolutionalLayer(8, 20, 20, 2, 1, 2)
	if err != nil {
		return nil, err
	}

	net, err := NewSequentialNet(
		cl0,
		NewReLULayer(cl0.GetOutputSize()...),
		NewPoolLayer(cl0.GetOutputSize(), []int{2, 2, 1}),
		cl1,
		NewReLULayer(cl1.GetOutputSize()...),
		NewPoolLayer(cl1.GetOutputSize(), []int{2, 2, 1}),
		cl2,
		NewReLULayer(cl2.GetOutputSize()...),
		NewPoolLayer(cl2.GetOutputSize(), []int{2, 2, 1}),
		NewDenseLayer([]int{4, 4, 20}, []int{10}),
		NewSoftmaxLayer(10),
	)

	if err != nil {
		return nil, err
	}

	return net, nil

}

func BenchmarkCIFAR10ConvNetDemoActivation(b *testing.B) {
	net, err := createCIFAR10ConvNetJSDemoNet()

	if err != nil {
		b.Fatalf(err.Error())
	}

	benchmarkConvNetActivation(b, 32, 3, net)

}

func BenchmarkCIFAR10ConvNetDemoBackPropagation(b *testing.B) {
	net, err := createCIFAR10ConvNetJSDemoNet()

	if err != nil {
		b.Fatalf(err.Error())
	}

	benchmarkConvNetBP(b, 32, 3, net)
}

func benchmarkConvNetActivation(b *testing.B, size int, depth int, net weight.Layer) {
	prev := runtime.GOMAXPROCS(1)

	arr := make([]float64, size*size*depth)
	for j := 0; j < len(arr); j++ {
		arr[j] = rand.Float64()
	}

	input := &tensor.Tensor{Size: []int{size, size, depth}, Values: arr}

	for i := 0; i < b.N; i++ {
		_, err := net.Activate(input)
		if err != nil {
			b.Fatalf(err.Error())
		}
	}

	runtime.GOMAXPROCS(prev)
}

func benchmarkConvNetBP(b *testing.B, size int, depth int, net weight.BPLearnerLayer) {
	prev := runtime.GOMAXPROCS(1)
	arrGrad := make([]float64, 10)
	for j := 0; j < len(arrGrad); j++ {
		arrGrad[j] = rand.Float64()
	}

	arrIn := make([]float64, size*size*depth)
	for j := 0; j < len(arrIn); j++ {
		arrIn[j] = rand.Float64()
	}

	grad := &tensor.Tensor{Size: []int{10}, Values: arrGrad}
	input := &tensor.Tensor{Size: []int{size, size, depth}, Values: arrIn}

	var err error

	_, err = net.Activate(input)
	if err != nil {
		b.Fatalf(err.Error())
	}

	for i := 0; i < b.N; i++ {

		_, err = net.BackPropagate(grad)
		if err != nil {
			b.Fatalf(err.Error())
		}
	}
	runtime.GOMAXPROCS(prev)
}
