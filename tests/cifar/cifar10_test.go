package cifar

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/gerardabello/weight"
	"github.com/gerardabello/weight/augmentation"
	"github.com/gerardabello/weight/costs"
	"github.com/gerardabello/weight/debug"
	"github.com/gerardabello/weight/layers"
	"github.com/gerardabello/weight/loaders/cifar"
	"github.com/gerardabello/weight/training"
)

func TestCIFAR10Open(t *testing.T) {
	base := `./cifar-10-batches-bin/`

	_, err := cifar.NewCIFAR10Set([]string{
		base + `data_batch_1.bin`,
		base + `data_batch_2.bin`,
		base + `data_batch_3.bin`,
		base + `data_batch_4.bin`,
		base + `data_batch_5.bin`,
	})

	if err != nil {
		t.Fatal(err)
	}
}

//https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
//Forward time per example: ~8ms
//Backprop time per example: ~11ms
func TestCIFAR10ConvNetDemo(t *testing.T) {

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	cl0 := layers.NewSquareConvolutionalLayer(32, 3, 16, 2, 1, 2)

	cl1 := layers.NewSquareConvolutionalLayer(16, 16, 20, 2, 1, 2)

	cl2 := layers.NewSquareConvolutionalLayer(8, 20, 20, 2, 1, 2)

	net, err := layers.NewSequentialNet(
		layers.NewReshaperLayer([]int{32, 32, 3}, []int{32, 32, 3}),
		cl0,
		layers.NewReLULayer(cl0.GetOutputSize()...),
		layers.NewPoolLayer(cl0.GetOutputSize(), []int{2, 2, 1}),
		cl1,
		layers.NewReLULayer(cl1.GetOutputSize()...),
		layers.NewPoolLayer(cl1.GetOutputSize(), []int{2, 2, 1}),
		cl2,
		layers.NewReLULayer(cl2.GetOutputSize()...),
		layers.NewPoolLayer(cl2.GetOutputSize(), []int{2, 2, 1}),
		layers.NewDenseLayer([]int{4, 4, 20}, []int{600}),
		layers.NewDenseLayer([]int{600}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	config := training.LearningConfig{BatchSize: 16, Epochs: 100, WeightDecay: 0.0001, Method: training.AdaDelta}

	testLearnCIFAR10(t, net, config, 0.60)

}

//They say they get arround 86% with local response normalization
func TestCIFAR10TFTutorial(t *testing.T) {

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	cl0 := layers.NewSquareConvolutionalLayer(24, 3, 64, 2, 1, 2)

	cl1 := layers.NewSquareConvolutionalLayer(12, 64, 64, 2, 1, 2)

	net, err := layers.NewSequentialNet(
		layers.NewReshaperLayer([]int{24, 24, 3}, []int{24, 24, 3}),
		cl0,
		layers.NewLeakyReLULayer(cl0.GetOutputSize()...),
		layers.NewPoolLayer(cl0.GetOutputSize(), []int{2, 2, 1}),
		//LRN
		cl1,
		layers.NewLeakyReLULayer(cl1.GetOutputSize()...),
		//LRN
		layers.NewPoolLayer(cl1.GetOutputSize(), []int{2, 2, 1}),
		layers.NewDenseLayer([]int{6, 6, 64}, []int{384}),
		layers.NewDenseLayer([]int{384}, []int{192}),
		layers.NewDenseLayer([]int{192}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	config := training.LearningConfig{BatchSize: 128, Epochs: 600, WeightDecay: 0.0001, LearningRateStart: 0.1, LearningRateEnd: 0.001, Momentum: 0, Method: training.Momentum}

	testLearnCIFAR10Cropped(t, net, config, 0.80)

}

func TestCIFAR10LeNet(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	cl0 := layers.NewSquareConvolutionalLayer(32, 3, 20, 2, 1, 2)

	cl1 := layers.NewSquareConvolutionalLayer(16, 20, 20, 2, 1, 2)

	net, err := layers.NewSequentialNet(
		cl0,
		layers.NewReLULayer(cl0.GetOutputSize()...),
		layers.NewPoolLayer(cl0.GetOutputSize(), []int{2, 2, 1}),
		cl1,
		layers.NewReLULayer(cl1.GetOutputSize()...),
		layers.NewPoolLayer(cl1.GetOutputSize(), []int{2, 2, 1}),
		layers.NewDenseLayer([]int{8, 8, 20}, []int{500}),
		layers.NewReLULayer(500),
		layers.NewDenseLayer([]int{500}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	config := training.LearningConfig{BatchSize: 8, Epochs: 30, WeightDecay: 0.0001, Method: training.AdaDelta}

	testLearnCIFAR10(t, net, config, 0.60)

}

func TestCIFAR10CaffeQuick(t *testing.T) {

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	cl1 := layers.NewSquareConvolutionalLayer(32, 3, 32, 2, 1, 2)

	cl2 := layers.NewSquareConvolutionalLayer(16, 32, 32, 2, 1, 2)

	cl3 := layers.NewSquareConvolutionalLayer(8, 32, 64, 2, 1, 2)

	net, err := layers.NewSequentialNet(
		cl1,
		layers.NewReLULayer(cl1.GetOutputSize()...),
		layers.NewPoolLayer(cl1.GetOutputSize(), []int{2, 2, 1}),
		cl2,
		layers.NewReLULayer(cl2.GetOutputSize()...),
		layers.NewPoolLayer(cl2.GetOutputSize(), []int{2, 2, 1}),
		cl3,
		layers.NewReLULayer(cl3.GetOutputSize()...),
		layers.NewPoolLayer(cl3.GetOutputSize(), []int{2, 2, 1}),
		layers.NewDenseLayer([]int{4, 4, 64}, []int{600}),
		layers.NewDenseLayer([]int{600}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	config := training.LearningConfig{BatchSize: 100, Epochs: 100, LearningRateStart: 0.0001, LearningRateEnd: 0.0001, WeightDecay: 0.00001, Method: training.Adam}

	testLearnCIFAR10(t, net, config, 0.70)

}
func TestCIFAR10Dense(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	rand.Seed(4)

	hln1 := 50

	net, err := layers.NewSequentialNet(
		layers.NewDenseLayer([]int{32, 32, 3}, []int{hln1}),
		layers.NewSigmoidLayer(hln1),
		layers.NewDenseLayer([]int{hln1}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	testLearnCIFAR10(t, net, training.LearningConfig{BatchSize: 16, Epochs: 100, WeightDecay: 0.0001, Method: training.AdaDelta}, 0.80)

}

func testLearnCIFAR10(t *testing.T, net weight.BPLearnerLayer, config training.LearningConfig, targetAccuracy float64) {
	assert := assert.New(t)

	base := `./cifar-10-batches-bin/`

	data, err := cifar.OpenCIFAR10(base)

	if err != nil {
		t.Fatalf(err.Error())
	}
	defer data.Close()

	trainer := training.NewBPTrainer(config, data, net, costs.NewCrossEntropyCostFunction(10))
	trainer.SetDebugger(&debug.HttpDebugger{})
	err = trainer.Train()
	if err != nil {
		t.Fatalf(err.Error())
	}

	accuracy, err := weight.TestLayer(net, data.TestSet)
	if err != nil {
		t.Fatalf(err.Error())
	}

	t.Logf("Acuracy: %.4f", accuracy)

	assert.True(accuracy > targetAccuracy, fmt.Sprintf("Expected accuracy bigger than %f", targetAccuracy))
}

func testLearnCIFAR10Cropped(t *testing.T, net weight.BPLearnerLayer, config training.LearningConfig, targetAccuracy float64) {
	assert := assert.New(t)

	base := `./cifar-10-batches-bin/`

	data, err := cifar.OpenCIFAR10(base)
	data.TrainSet = &augmentation.Cropper{DataSet: data.TrainSet, MaxAmount: []int{4, 4, 0}}
	data.TestSet = &augmentation.Cropper{DataSet: data.TestSet, MaxAmount: []int{4, 4, 0}}

	if err != nil {
		t.Fatalf(err.Error())
	}
	defer data.Close()

	trainer := training.NewBPTrainer(config, data, net, costs.NewCrossEntropyCostFunction(10))
	trainer.SetDebugger(&debug.HttpDebugger{})
	err = trainer.Train()
	if err != nil {
		t.Fatalf(err.Error())
	}

	accuracy, err := weight.TestLayer(net, data.TestSet)
	if err != nil {
		t.Fatalf(err.Error())
	}

	t.Logf("Acuracy: %.4f", accuracy)

	assert.True(accuracy > targetAccuracy, fmt.Sprintf("Expected accuracy bigger than %f", targetAccuracy))
}
