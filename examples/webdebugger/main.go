package main

import (
	"github.com/gerardabello/weight"
	"github.com/gerardabello/weight/costs"
	"github.com/gerardabello/weight/debug"
	"github.com/gerardabello/weight/layers"
	"github.com/gerardabello/weight/loaders/cifar"
	"github.com/gerardabello/weight/training"
)

func main() {

	net := createTestNet()

	//Setup learning configuration
	config := training.LearningConfig{
		BatchSize:   20,
		Epochs:      20,
		WeightDecay: 0.0001,
		Method:      training.AdaDelta,
	}

	base := `./cifar-10-batches-bin/`

	data, err := cifar.OpenCIFAR10(base)
	if err != nil {
		panic(err)
	}
	defer data.Close()

	//New a cost function. As we want to classify and we have a softmax as the last layer, we use a cross entropy function with 10 inputs/outputs.
	costFunc := costs.NewCrossEntropyCostFunction(10)

	//New a trainer
	trainer := training.NewBPTrainer(config, data, net, costFunc)

	//Set custom debugger
	trainer.SetDebugger(&debug.HttpDebugger{})

	//Start training. As we have config.DebugPrint to true, we should see training data in the standard output. As we also have config.TestEveryEpoch to true, the trainer will calculate the accuracy with the test data after every epoch.
	err = trainer.Train()
	if err != nil {
		panic(err)
	}
}

func createTestNet() weight.BPLearnerLayer {
	//New a layer with 3 convolutional layers (16 kernels each) and 2 dense layers (300 neurons each). Input size 28x28 and 10 labels, as we are going to use mnist dataset.
	net := layers.NewCRPBlocks([]int{32, 32, 2}, 3, 12, 3)

	return net
}
