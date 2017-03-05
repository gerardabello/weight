package weight

import "github.com/gerardabello/weight/tensor"

//Layer is any object that accepts tensor.Tensor and returns tensor.Tensor.
//Each tensor.Tensor has a shape specified by GetSize. Layers only accept a specific shape. If any shape does not match the requirements the Layer will return error on Activate.
type Layer interface {
	ID() string

	//Activate takes and input tensor and computes an output tensor given the parameters and configuration of the layer
	Activate(input *tensor.Tensor) (*tensor.Tensor, error)
	GetInputSize() []int
	GetOutputSize() []int
}

//EnslaverLayer is a layer than can create slaves of itself
type EnslaverLayer interface {
	//TODO maybe merge this with BPLearnerLayer
	Layer

	//A slave is a copy of the layer but with the parameters (for example weghts & biases) as a pointer to the parameters of the original layer. This is used to learn a batch in parallel. The learning method should be able to call different slave's Activate and Backpropagate methods in parallel. The parameter's update will be done syncronously, only on the base layer using the gradients from all slave layers.
	CreateSlave() Layer
}

//BPLearnerLayer is a layer that can be trained using backpropagation
type BPLearnerLayer interface {
	Layer

	//BackPropagate updates the stored gradient for each parameter and returns the backpropagated gradient
	BackPropagate(err *tensor.Tensor) (*tensor.Tensor, error)

	//GetParamGradPointers returns a slice of pointers to prameters and gradients (in the same order) so gradient descent can update them.
	GetParamGradPointers() ([]*float64, []*float64)
}
