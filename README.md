# Weight
Weight is a neural network library written in Go that focuses on portability and ease of use.

The library has three main parts:
* The `Tensor` struct
* Interface definitions
* Interface implementations

Appart from the Tensor struct, everything is an interface, so you can extend/replace the default implementation with your own.

## Usage

Let's see how to classify digits from the MNIST dataset using this library:

The first thing we need is a layer. A layer takes a tensor and returns another one, usually by computing some mathematical operation.
A network is a layer that contains other layers. The most basic type of network is one that activates layers sequentially. Most problems are too complex for only one layer so almost always you will use a network.

The network must take an input image of 28x28 pixels and return 10 values, one for each possible digit.

```go
net, _ := layers.NewSequentialNet(
    layers.NewDenseLayer([]int{28, 28}, []int{30}),  //Dense layer with input size 28x28 and output size 30
    layers.NewSigmoidLayer(30),                      //As a sigmoid layer has the same input and output size, we just define one (30)
    layers.NewDenseLayer([]int{30}, []int{10}),
    layers.NewSoftmaxLayer(10),
)
```

We also need data, in the form of a struct that implements the DataSet interface. Weight includes some implementations for MNIST, CIFAR, etc.
You probably need to implement this interface to fit the needs of your data. See `weight/loaders` for example implementations.

Most loaders return a PairSet, that's just two DataSet structs: one for training and one for testing.

```go
//Parse the mnist files from the current folder and return a PairSet
data, _ := mnist.Open(".")
```

What we want to do now is to train the network. We must first define two things: how to evaluate it's output and what learning method to use.

To evaluate the network's output we must use a cost function. It is a function that takes the output of the network for a given data point and the correct answer. It outputs a value that represents how incorrect the output from the network was.


In this case we use a cross entropy cost function of size 10, the same as the output of the net.
```go
costFunc := costs.NewCrossEntropyCostFunction(10)
```

Weight uses gradient descent to train the network, but there are different methods and parameters to choose from.
You can see an overview of them here: http://sebastianruder.com/optimizing-gradient-descent/

In this case we will use momentum with a learning rate that decays from 0.5 to 0.1 for 5 epochs (one epoch is a run through all data points in the train set).

```go
config := training.LearningConfig{
    BatchSize:         16,
    Epochs:            5,
    Method:            training.Momentum,
    LearningRateStart: 0.5,
    LearningRateEnd:   0.1,
    Momentum:          0.9,
}
```

The only thing that we need now is to create a trainer with all the information and start training.

```go
//Create a trainer with the network, configuration, cost function and data.
trainer := training.NewBPTrainer(config, data, net, costFunc)

//Start training.
_ = trainer.Train()
```

It is important to test the network after training. The test set contains images that are not in the train set, so it is good to test how it will perform on unknown images.
The method `TestLayer` returns the accuracy of the network from 0 to 1, 1 beeing the perfect score.

```go
//Get final accuracy on test data
accuracy, _ := weight.TestLayer(net, data.TestSet)
```

You can find and run the full example code in `examples/readme`.
The training takes some seconds and the final accuracy should be around 92%. This result is really bad for MNIST, but with a convolutional neural network we can achieve close to 99%.


## Layers implemented
* FFNet (Feed forward network)
* Convolutional (3D)
* Dense/Fully connected
* Pool
* ReLU
* LeakyReLU
* Sigmoid
* Softmax

## TODO
* Add way to save and load networks (marshaling)
* Add GPU computations
* Allow to configure initialization of parameters
* Compute backpropagation using col2im
* More unit testing
* Recurrent layers
