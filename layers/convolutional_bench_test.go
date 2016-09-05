package layers

import (
	"math/rand"
	"testing"

	"gitlab.com/gerardabello/weight/tensor"
)

func benchmarkConvLayerActivation(b *testing.B, size int, depth int) {
	arr := make([]float64, size*size)
	for j := 0; j < len(arr); j++ {
		arr[j] = rand.Float64()
	}

	input := &tensor.Tensor{Size: []int{size, size, 1}, Values: arr}

	cl0, err := NewSquareConvolutionalLayer(size, 1, depth, 2, 1, 2)
	if err != nil {
		b.Fatalf(err.Error())
	}

	net := cl0

	if err != nil {
		b.Fatalf(err.Error())
	}

	for i := 0; i < b.N; i++ {
		_, err := net.Activate(input)
		if err != nil {
			b.Fatalf(err.Error())
		}
	}
}

func BenchmarkConvLayerActivation10_10(b *testing.B)   { benchmarkConvLayerActivation(b, 10, 10) }
func BenchmarkConvLayerActivation10_100(b *testing.B)  { benchmarkConvLayerActivation(b, 10, 100) }
func BenchmarkConvLayerActivation10_1000(b *testing.B) { benchmarkConvLayerActivation(b, 10, 1000) }
func BenchmarkConvLayerActivation100_10(b *testing.B)  { benchmarkConvLayerActivation(b, 100, 10) }
func BenchmarkConvLayerActivation1000_10(b *testing.B) { benchmarkConvLayerActivation(b, 1000, 10) }

func benchmarkConvLayerBP(b *testing.B, size int, depth int) {
	arr := make([]float64, size*size*depth)
	for j := 0; j < len(arr); j++ {
		arr[j] = rand.Float64()
	}

	grad := &tensor.Tensor{Size: []int{size, size, depth}, Values: arr}
	input := &tensor.Tensor{Size: []int{size, size, 1}, Values: arr}

	cl0, err := NewSquareConvolutionalLayer(size, 1, depth, 2, 1, 2)
	if err != nil {
		b.Fatalf(err.Error())
	}

	net, err := NewSequentialNet(
		cl0,
	)
	if err != nil {
		b.Fatalf(err.Error())
	}

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
}

func BenchmarkConvLayerBP10_10(b *testing.B)   { benchmarkConvLayerBP(b, 10, 10) }
func BenchmarkConvLayerBP10_100(b *testing.B)  { benchmarkConvLayerBP(b, 10, 100) }
func BenchmarkConvLayerBP10_1000(b *testing.B) { benchmarkConvLayerBP(b, 10, 1000) }
func BenchmarkConvLayerBP100_10(b *testing.B)  { benchmarkConvLayerBP(b, 100, 10) }
func BenchmarkConvLayerBP1000_10(b *testing.B) { benchmarkConvLayerBP(b, 1000, 10) }
