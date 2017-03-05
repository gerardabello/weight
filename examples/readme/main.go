package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/gerardabello/weight"
	"github.com/gerardabello/weight/costs"
	"github.com/gerardabello/weight/layers"
	"github.com/gerardabello/weight/loaders/mnist"
	"github.com/gerardabello/weight/training"
)

func main() {
	println("Downloading MNIST dataset. This can take some time...")
	//Download dataset if not exists
	download(`http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz`)
	download(`http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz`)
	download(`http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz`)
	download(`http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz`)

	//Create a simple neural network.
	//The output size from one network has to be equal to the input size of the next. In this case (28x28)->30->10.
	net, err := layers.NewSequentialNet(
		//The input to the first layer is the image size (28x28)
		layers.NewDenseLayer([]int{28, 28}, []int{30}),
		//We use 30 hidden neurons
		layers.NewSigmoidLayer(30),
		layers.NewDenseLayer([]int{30}, []int{10}),
		//The output is the number of possible digits
		layers.NewSoftmaxLayer(10),
	)

	if err != nil {
		panic("Error creating network: " + err.Error())
	}

	//Setup learning configuration.
	//You can find an overview of the different methods at http://sebastianruder.com/optimizing-gradient-descent/.
	config := training.LearningConfig{
		BatchSize:         16,
		Epochs:            5,
		Method:            training.Momentum,
		LearningRateStart: 0.5,
		LearningRateEnd:   0.1,
		Momentum:          0.9,
	}

	//Open path where we have the mnist data files.
	//mnist.Open returns a PairSet, that contains the TrainSet and the corresponding TestSet
	data, err := mnist.Open(".")
	if err != nil {
		panic("Error opening dataset: " + err.Error())
	}
	defer data.Close()

	//Create a cost function. As we want to classify and we have a softmax as the last layer, we use a cross entropy function.
	//The size has to be the same as the output of the network.
	costFunc := costs.NewCrossEntropyCostFunction(10)

	//Create a trainer with the network, configuration, cost function and data.
	trainer := training.NewBPTrainer(config, data, net, costFunc)

	//Start training.
	err = trainer.Train()
	if err != nil {
		panic("Error training network: " + err.Error())
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
