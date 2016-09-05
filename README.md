# Weight
Weight is a neural network library written in Go that focuses on portability and ease of use. At the moment is not production ready.

## Example usage

Here is an example of a simple network with one hidden layer of 30 neurons. Note that this library does not include the MNIST data files but you can download them from http://yann.lecun.com/exdb/mnist/.

```go
//TODO include code from examples/readme  
```

After some seconds it should have calculated all 5 epochs and the test accuracy should be around 92%. This result is really bad for MNIST, but with a convolutional neural network we can achieve close to 99%.

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
* Automatically download common datasets like MNIST or CIFAR
* Compute backpropagation using col2im
* More unit testing
