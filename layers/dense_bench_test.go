package layers

import (
	"math/rand"
	"testing"

	"github.com/gerardabello/weight/tensor"
)

func benchmarkDenseActivation(size int, b *testing.B) {
	layer := NewDenseLayer([]int{size}, []int{size})

	input := make([]float64, size)
	for i := 0; i < size; i++ {
		input[i] = rand.Float64()
	}
	data := &tensor.Tensor{Size: []int{size}, Values: input}

	for n := 0; n < b.N; n++ {
		_, err := layer.Activate(data)

		if err != nil {
			b.Errorf("Error while activating layer: %s", err.Error())
		}
	}
}

func BenchmarkDenseActivation10(b *testing.B)    { benchmarkDenseActivation(10, b) }
func BenchmarkDenseActivation100(b *testing.B)   { benchmarkDenseActivation(100, b) }
func BenchmarkDenseActivation1000(b *testing.B)  { benchmarkDenseActivation(1000, b) }
func BenchmarkDenseActivation10000(b *testing.B) { benchmarkDenseActivation(10000, b) }

func benchmarkDenseBP(size int, b *testing.B) {
	layer := NewDenseLayer([]int{size}, []int{size})

	input := make([]float64, size)
	for i := 0; i < size; i++ {
		input[i] = rand.Float64()
	}
	data := &tensor.Tensor{Size: []int{size}, Values: input}

	out, err := layer.Activate(data)

	if err != nil {
		b.Errorf("Error while activating layer: %s", err.Error())
	}

	for n := 0; n < b.N; n++ {
		_, err := layer.BackPropagate(out)

		if err != nil {
			b.Errorf("Error while backpropagating layer: %s", err.Error())
		}
	}
}

func BenchmarkDenseBP10(b *testing.B)    { benchmarkDenseBP(10, b) }
func BenchmarkDenseBP100(b *testing.B)   { benchmarkDenseBP(100, b) }
func BenchmarkDenseBP1000(b *testing.B)  { benchmarkDenseBP(1000, b) }
func BenchmarkDenseBP10000(b *testing.B) { benchmarkDenseBP(10000, b) }
