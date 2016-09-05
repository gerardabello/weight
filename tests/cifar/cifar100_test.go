package cifar

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/costs"
	"gitlab.com/gerardabello/weight/layers"
	"gitlab.com/gerardabello/weight/loaders/cifar"
	"gitlab.com/gerardabello/weight/training"
)

//https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
func TestCIFAR100ConvNetDemo(t *testing.T) {
	cl0, err := layers.NewSquareConvolutionalLayer(32, 3, 16, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	cl1, err := layers.NewSquareConvolutionalLayer(16, 16, 20, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	cl2, err := layers.NewSquareConvolutionalLayer(8, 20, 20, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	net, err := layers.NewSequentialNet(
		cl0,
		layers.NewReLULayer(cl0.GetOutputSize()...),
		layers.NewPoolLayer(cl0.GetOutputSize(), []int{2, 2, 1}),
		cl1,
		layers.NewReLULayer(cl1.GetOutputSize()...),
		layers.NewPoolLayer(cl1.GetOutputSize(), []int{2, 2, 1}),
		cl2,
		layers.NewReLULayer(cl2.GetOutputSize()...),
		layers.NewPoolLayer(cl2.GetOutputSize(), []int{2, 2, 1}),
		layers.NewDenseLayer([]int{4, 4, 20}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	config := training.LearningConfig{BatchSize: 4, Epochs: 100, WeightDecay: 0.0001, Method: training.AdaDelta}

	testLearnCIFAR100(t, net, config, 0.60)

}

func TestCIFAR100LeNet(t *testing.T) {
	cl0, err := layers.NewSquareConvolutionalLayer(32, 3, 20, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	cl1, err := layers.NewSquareConvolutionalLayer(16, 20, 20, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

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

	testLearnCIFAR100(t, net, config, 0.60)

}

func TestCIFAR100Convolutional(t *testing.T) {
	cl0, err := layers.NewSquareConvolutionalLayer(32, 3, 16, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	cl1, err := layers.NewSquareConvolutionalLayer(16, 16, 16, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	net, err := layers.NewSequentialNet(
		cl0,
		layers.NewReLULayer(cl0.GetOutputSize()...),
		layers.NewPoolLayer(cl0.GetOutputSize(), []int{2, 2, 1}),
		cl1,
		layers.NewReLULayer(cl1.GetOutputSize()...),
		layers.NewPoolLayer(cl1.GetOutputSize(), []int{2, 2, 1}),
		layers.NewDenseLayer([]int{8, 8, 16}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	config := training.LearningConfig{BatchSize: 4, Epochs: 30, WeightDecay: 0.0001, Method: training.AdaDelta}

	testLearnCIFAR100(t, net, config, 0.50)

}
func TestCIFAR100Dense(t *testing.T) {
	//rand.Seed(time.Now().UnixNano())
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

	testLearnCIFAR100(t, net, training.LearningConfig{LearningRateStart: 0.5, LearningRateEnd: 0.01, BatchSize: 4, Epochs: 5, WeightDecay: 0}, 0.80)

}

func testLearnCIFAR100(t *testing.T, net weight.BPLearnerLayer, config training.LearningConfig, targetAccuracy float64) {
	assert := assert.New(t)

	base := `./cifar-100-binary/`

	data, err := cifar.OpenCIFAR100(base)
	if err != nil {
		t.Fatalf(err.Error())
	}
	defer data.Close()

	trainer := training.NewBPTrainer(config, data, net, costs.NewCrossEntropyCostFunction(10))
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
