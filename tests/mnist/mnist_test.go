package mnist

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/costs"
	"gitlab.com/gerardabello/weight/debug"
	"gitlab.com/gerardabello/weight/layers"
	"gitlab.com/gerardabello/weight/loaders/mnist"
	"gitlab.com/gerardabello/weight/training"
)

func TestMNISTLeNet1Fast(t *testing.T) {

	net, err := layers.NewSequentialNet(
		layers.NewReshaperLayer([]int{28, 28}, []int{28, 28, 1}),
		layers.NewCRPBlock([]int{28, 28, 1}, 1, 8),
		layers.NewDenseLayer([]int{14, 14, 8}, []int{200}),
		layers.NewReLULayer(200),
		layers.NewDenseLayer([]int{200}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	testLearnMNIST(t, net, training.LearningConfig{BatchSize: 12, Epochs: 2, LearningRateStart: 0.001, LearningRateEnd: 0.0001, WeightDecay: 0.0001, Method: training.Adam}, 0.96)
}

func TestMNISTLeNet2(t *testing.T) {

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	net, err := layers.NewSequentialNet(
		layers.NewReshaperLayer([]int{28, 28}, []int{28, 28, 1}),
		layers.NewCRPBlocks([]int{28, 28, 1}, 1, 20, 2),
		layers.NewDenseLayer([]int{7, 7, 20}, []int{500}),
		layers.NewReLULayer(500),
		layers.NewDenseLayer([]int{500}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	testLearnMNIST(t, net, training.LearningConfig{BatchSize: 32, Epochs: 4, LearningRateStart: 0.001, LearningRateEnd: 0.0001, WeightDecay: 0.0001, Method: training.Adam}, 0.985)
}

func TestMNISTLeNet3(t *testing.T) {

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	cl0, err := layers.NewSquareConvolutionalLayer(28, 1, 16, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	cl1, err := layers.NewSquareConvolutionalLayer(14, 16, 32, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	cl2, err := layers.NewSquareConvolutionalLayer(7, 32, 32, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	net, err := layers.NewSequentialNet(
		layers.NewReshaperLayer([]int{28, 28}, []int{28, 28, 1}),
		cl0,
		layers.NewReLULayer(cl0.GetOutputSize()...),
		layers.NewPoolLayer(cl0.GetOutputSize(), []int{2, 2, 1}),
		cl1,
		layers.NewReLULayer(cl1.GetOutputSize()...),
		layers.NewPoolLayer(cl1.GetOutputSize(), []int{2, 2, 1}),
		cl2,
		layers.NewReLULayer(cl2.GetOutputSize()...),
		layers.NewDenseLayer([]int{7, 7, 32}, []int{300}),
		layers.NewReLULayer(300),
		layers.NewDenseLayer([]int{300}, []int{100}),
		layers.NewReLULayer(100),
		layers.NewDenseLayer([]int{100}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	testLearnMNIST(t, net, training.LearningConfig{BatchSize: 32, Epochs: 20, WeightDecay: 0.0001, Method: training.AdaDelta}, 0.99)
}

func TestMNISTConvTF(t *testing.T) {
	//Try to copy TensorFlow's MNIST tutorial. They claim TF gets 99.2% with this net.
	// TF computes a batch of 50 in about 100ms (460 SPS) in my PC
	// weight computes a batch of 48 in about 1000ms (45 SPS) in my PC
	//TODO improve this

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	cl0, err := layers.NewSquareConvolutionalLayer(28, 1, 32, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	cl1, err := layers.NewSquareConvolutionalLayer(14, 32, 64, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	net, err := layers.NewSequentialNet(
		layers.NewReshaperLayer([]int{28, 28}, []int{28, 28, 1}),
		cl0,
		layers.NewReLULayer(cl0.GetOutputSize()...),
		layers.NewPoolLayer(cl0.GetOutputSize(), []int{2, 2, 1}),
		cl1,
		layers.NewReLULayer(cl1.GetOutputSize()...),
		layers.NewPoolLayer(cl1.GetOutputSize(), []int{2, 2, 1}),
		layers.NewDenseLayer([]int{7, 7, 64}, []int{1024}),
		layers.NewReLULayer(1024),
		layers.NewDenseLayer([]int{1024}, []int{10}),
		//Dropout
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	//lr: 0.5->0.01  batch:4, epochs:5, decay:0  accuracy:0.9839
	testLearnMNIST(t, net, training.LearningConfig{BatchSize: 48, Epochs: 17, WeightDecay: 0.0001, Method: training.AdaDelta}, 0.99)
}

func TestMNISTCaffe(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	cl0, err := layers.NewSquareConvolutionalLayer(28, 1, 20, 2, 1, 0)
	if err != nil {
		t.Fatalf(err.Error())
	}

	net, err := layers.NewSequentialNet(
		layers.NewReshaperLayer([]int{28, 28}, []int{28, 28, 1}),
		cl0,
		layers.NewPoolLayer(cl0.GetOutputSize(), []int{2, 2, 1}),
		layers.NewDenseLayer([]int{12, 12, 20}, []int{500}),
		layers.NewReLULayer(500),
		layers.NewDenseLayer([]int{500}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	testLearnMNIST(t, net, training.LearningConfig{BatchSize: 60, Epochs: 10, WeightDecay: 0.0005, Momentum: 0.9, LearningRateStart: 0.01, LearningRateEnd: 0, Method: training.Momentum}, 0.98)
}

//try to emulate http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html
func TestMNISTConvNetDemo(t *testing.T) {

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	cl0, err := layers.NewSquareConvolutionalLayer(28, 1, 8, 2, 1, 0)
	if err != nil {
		t.Fatalf(err.Error())
	}

	cl1, err := layers.NewSquareConvolutionalLayer(12, 8, 16, 2, 1, 2)
	if err != nil {
		t.Fatalf(err.Error())
	}

	net, err := layers.NewSequentialNet(
		layers.NewReshaperLayer([]int{28, 28}, []int{28, 28, 1}),
		cl0,
		layers.NewReLULayer(cl0.GetOutputSize()...),
		layers.NewPoolLayer(cl0.GetOutputSize(), []int{2, 2, 1}),

		cl1,
		layers.NewReLULayer(cl1.GetOutputSize()...),
		layers.NewPoolLayer(cl1.GetOutputSize(), []int{3, 3, 1}),

		layers.NewDenseLayer([]int{4, 4, 16}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	testLearnMNIST(t, net, training.LearningConfig{BatchSize: 20, Epochs: 10, WeightDecay: 0.001, Method: training.AdaDelta}, 0.98)
}
func TestMNISTDense(t *testing.T) {

	//rand.Seed(time.Now().UnixNano())
	rand.Seed(4)

	hln1 := 50

	net, err := layers.NewSequentialNet(
		layers.NewDenseLayer([]int{28, 28}, []int{hln1}),
		layers.NewSigmoidLayer(hln1),
		layers.NewDenseLayer([]int{hln1}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	config := training.LearningConfig{
		BatchSize:         12,
		Epochs:            5,
		LearningRateStart: 0.5,
		LearningRateEnd:   0.1,
		Momentum:          0.9,
		Method:            training.Momentum,
	}
	testLearnMNIST(t, net, config, 0.93)
}

func TestMNISTLinear(t *testing.T) {

	//rand.Seed(time.Now().UnixNano())
	rand.Seed(4)

	net, err := layers.NewSequentialNet(
		layers.NewDenseLayer([]int{28, 28}, []int{10}),
		layers.NewSigmoidLayer(10),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		t.Fatalf(err.Error())
	}

	config := training.LearningConfig{BatchSize: 12, Epochs: 10, LearningRateStart: 0.001, LearningRateEnd: 0.001, Method: training.Adam}
	testLearnMNIST(t, net, config, 0.91)
}

func TestMNISTFullyDeep(t *testing.T) {

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	rand.Seed(4)

	hln1 := 80
	hln2 := 70
	hln3 := 60
	hln4 := 50
	hln5 := 40

	net, err := layers.NewSequentialNet(
		layers.NewDenseLayer([]int{28, 28}, []int{hln1}),
		layers.NewReLULayer(hln1),
		layers.NewDenseLayer([]int{hln1}, []int{hln2}),
		layers.NewReLULayer(hln2),
		layers.NewDenseLayer([]int{hln2}, []int{hln3}),
		layers.NewReLULayer(hln3),
		layers.NewDenseLayer([]int{hln3}, []int{hln4}),
		layers.NewReLULayer(hln4),
		layers.NewDenseLayer([]int{hln4}, []int{hln5}),
		layers.NewReLULayer(hln5),
		layers.NewDenseLayer([]int{hln5}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		println(err.Error())
	}

	config := training.LearningConfig{LearningRateStart: 0.3, LearningRateEnd: 0.005, BatchSize: 4, Epochs: 20, Momentum: 0.5, WeightDecay: 0.00001}

	testLearnMNIST(t, net, config, 0.97)
}

func testLearnMNIST(t *testing.T, net weight.BPLearnerLayer, config training.LearningConfig, targetAccuracy float64) {
	assert := assert.New(t)

	//MNIST
	data, err := mnist.Open(".")
	if err != nil {
		t.Fatalf(err.Error())
	}
	defer data.Close()

	trainer := training.NewBPTrainer(config, data, net, costs.NewCrossEntropyCostFunction(10))
	trainer.SetDebugger(&debug.CLIDebugger{})
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
