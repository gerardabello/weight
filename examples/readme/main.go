package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/costs"
	"gitlab.com/gerardabello/weight/layers"
	"gitlab.com/gerardabello/weight/loaders/mnist"
	"gitlab.com/gerardabello/weight/training"
)

func main() {
	//Download dataset if not exists
	download(`http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz`)
	download(`http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz`)
	download(`http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz`)
	download(`http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz`)

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

	//Open path where we have the mnist data files (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz` and `t10k-labels-idx1-ubyte.gz`).
	//mnist.Open returns a PairSet, that contains the TrainSet and the corresponding TestSet
	data, err := mnist.Open(".")
	if err != nil {
		panic(err)
	}
	defer data.Close()

	//Create a cost function. As we want to classify and we have a softmax as the last layer, we use a cross entropy function. We use 10 inputs as we are classifying digits.
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

func download(url string) {
	tokens := strings.Split(url, "/")
	fileName := tokens[len(tokens)-1]

	if _, err := os.Stat(fileName); !os.IsNotExist(err) {
		return
	}

	fmt.Println("Downloading", url, "to", fileName)

	output, err := os.Create(fileName)
	if err != nil {
		fmt.Println("Error while creating", fileName, "-", err)
		return
	}
	defer output.Close()

	response, err := http.Get(url)
	if err != nil {
		fmt.Println("Error while downloading", url, "-", err)
		return
	}
	defer response.Body.Close()

	n, err := io.Copy(output, response.Body)
	if err != nil {
		fmt.Println("Error while downloading", url, "-", err)
		return
	}

	fmt.Println(n, "bytes downloaded.")
}
