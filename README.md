# Weight
Weight is a neural network library written in Go that focuses on portability and ease of use. At the moment is not production ready.

## Example usage

Here is an example of a simple network with one hidden layer of 30 neurons that we will train to classify digits from MNIST (http://yann.lecun.com/exdb/mnist/).

```go
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

//Open path where we have the mnist data files (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz` and `t10k-labels-idx1-ubyte.gz`). In the example in examples/readme, the files will be downloaded automatically.
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
```
You can find and run this code in `examples/readme`. It will download the MNIST dataset if it can't find it locally, and this can take some minutes.

The training takes some seconds and the final accuracy should be around 92%. This result is really bad for MNIST, but with a convolutional neural network we can achieve close to 99%.

Here we are using several layers, a dataset, a cost function and a trainer. All of these elements are interface implementations so it is really easy to create and use custom ones. The only structure you will provably need to implement to use this library in your project is a `DataSet`, the object that returns data and decides if the output of the network is correct or not (see loaders/mnist and loaders/cifar for example implementations of `DataSet`).

## Design
The primary design goal was to be able to implement and train a convolutional neural net in pure Go.

At the moment it has an implementation of a gradient decent trainer and several layers. The implementations are all CPU only, making this library relatively slow. GPU computations are not completely out of the picture, but at the moment the focus is on simplicity and portability.

The gradient descent trainer is multithreaded and includes regularization and different update methods like adagrad, adam, momentum, etc.

The layers implemented are:
* FFNet (Feed forward network)
* Convolutional (3D)
* Dense/Fully connected
* Pool
* ReLU
* LeakyReLU
* Sigmoid
* Softmax

Weight uses the struct `Tensor` to pass activations between layers. See `tensor/README.md` for more information.

## TODOs
* Add way to save and load networks (marshaling)
* Add GPU computations
* Allow to configure initialization of parameters
* Compute backpropagation using col2im
* More unit testing
