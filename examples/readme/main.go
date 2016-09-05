package main

import (
	"fmt"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/costs"
	"gitlab.com/gerardabello/weight/layers"
	"gitlab.com/gerardabello/weight/loaders/mnist"
	"gitlab.com/gerardabello/weight/training"
)

func main() {
	//Create a simple neural network (1 hidden layer)
	net, _ := layers.NewSequentialNet(
		layers.NewDenseLayer([]int{28, 28}, []int{30}),
		layers.NewSigmoidLayer(30),
		layers.NewDenseLayer([]int{30}, []int{10}),
		layers.NewSoftmaxLayer(10),
	)

	//Setup learning configuration
	config := training.LearningConfig{
		BatchSize:         16,
		Epochs:            5,
		LearningRateStart: 0.5,
		LearningRateEnd:   0.1,
		Momentum:          0.9,
		Method:            training.Momentum,
	}

	//Open path where we have the mnist data files (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz` and `t10k-labels-idx1-ubyte.gz`)
	data, err := mnist.Open(".")
	if err != nil {
		panic(err)
	}
	defer data.Close()

	//Create a cost function. As we want to classify and we have a softmax as the last layer, we use a cross entropy function with 10 inputs.
	costFunc := costs.NewCrossEntropyCostFunction(10)

	//Create a trainer. It is the object that will train the network with the given data and configuration.
	trainer := training.NewBPTrainer(config, data, net, costFunc)

	//Start training.
	err = trainer.Train()
	if err != nil {
		panic(err)
	}

	//Get final accuracy on test data
	accuracy, _ := weight.TestLayer(net, data.TestSet)
	fmt.Printf("Final accuracy: %.4f \n", accuracy)

}
