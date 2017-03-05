package layers

import (
	"errors"
	"fmt"

	"github.com/gerardabello/weight"
	"github.com/gerardabello/weight/debug"
	"github.com/gerardabello/weight/tensor"
)

//FFNode is a node to be used with FFNet
type FFNode struct {
	layer weight.Layer

	parents []*FFNode

	//Could be derived from the 'parents' array, but it would be slow to obtain them every time
	//Should be buffered channels of size 1
	inputs  []chan *tensor.Tensor //come from parents
	outputs []chan *tensor.Tensor //come from childs

	input         *tensor.Tensor //tensor to store the sum of incoming inputs on Activate()
	gradientError *tensor.Tensor //tensor to store the sum of incoming gradient errors on BackPropagate()
}

func (n *FFNode) ID() string {
	return n.layer.ID()
}

//Activate waits for the parent nodes to send its outputs, computes the sum of them and passes it to the underlying layer's Activate, then sends the result to all childs.
func (n *FFNode) Activate() {
	np := len(n.inputs)
	inputs := make([]*tensor.Tensor, np)

	for i := 0; i < np; i++ {
		inputs[i] = <-n.inputs[i]
		if !inputs[i].HasSize(n.layer.GetInputSize()) {
			panic(fmt.Sprintf("One of inputs to node %s has not the correct size. Actual:%v expected:%v \n", n.ID(), inputs[i].Size, n.layer.GetInputSize()))
		}
	}

	//Create a tensor if it doesn't exist
	if n.input == nil {
		n.input = tensor.NewTensor(n.layer.GetInputSize()...)
	} else {
		n.input.Zero(0)
	}

	err := n.input.Add(inputs...)
	if err != nil {
		panic(err)
	}

	//The undelying layer does the actual computation
	out, err := n.layer.Activate(n.input)
	if err != nil {
		panic(err)
	}

	//Send to all childs
	for i := 0; i < len(n.outputs); i++ {
		n.outputs[i] <- out
	}
}

//BackPropagate waits for the child nodes to send its propagated errors, computes the sum of them and passes it to the underlying layer's BackPropagate, then propagates the result to all parents.
func (n *FFNode) BackPropagate() {
	nc := len(n.outputs)

	gradientErrors := make([]*tensor.Tensor, nc)

	for i := 0; i < nc; i++ {
		gradientErrors[i] = <-n.outputs[i]
		if !gradientErrors[i].HasSize(n.layer.GetOutputSize()) {
			panic("One of errors to node has not the correct size")
		}
	}

	//Create a tensor if it doesn't exist
	if n.gradientError == nil {
		n.gradientError = tensor.NewTensor(n.layer.GetOutputSize()...)
	} else {
		n.gradientError.Zero(0)
	}

	err := n.gradientError.Add(gradientErrors...)
	if err != nil {
		panic(err)
	}

	//The undelying layer does the actual computation
	prop, err := n.layer.(weight.BPLearnerLayer).BackPropagate(n.gradientError)
	if err != nil {
		panic(err)
	}

	//Send to all parents
	for i := 0; i < len(n.inputs); i++ {
		n.inputs[i] <- prop
	}
}

//FFNet is a generic feedforward network. It can include any number branches, but they cannot form a loop.
type FFNet struct {
	id string

	input  chan *tensor.Tensor
	output chan *tensor.Tensor

	startNode *FFNode
	endNode   *FFNode

	nodes []*FFNode

	finished bool
}

//NewFFNet returns a new FFNet
func NewFFNet() *FFNet {
	net := &FFNet{}
	net.id = RandomID(8)
	return net
}

func (n *FFNet) ID() string {
	return n.id
}

//CreateSlave creates a slave of the FFNet. See EnslaverLayer in package weight for more information on layer slaves.
func (n *FFNet) CreateSlave() weight.Layer {
	ng := NewFFNet()

	ng.id = n.id

	for i := range n.nodes {
		var err error

		enslaver, ok := n.nodes[i].layer.(weight.EnslaverLayer)
		if !ok {
			panic("weight.Layer inside Sequential does not implement Enslaver interface")
		}

		parents := []string{}

		for _, parentNode := range n.nodes[i].parents {
			parents = append(parents, parentNode.ID())
		}

		err = ng.AddLayer(enslaver.CreateSlave(), parents...)
		if err != nil {
			panic(err)
		}

	}

	ng.End()

	return ng
}

//Activate takes an input tensor and passes it through all the layers in the netork following the node connections.
func (n *FFNet) Activate(input *tensor.Tensor) (*tensor.Tensor, error) {
	if !n.finished {
		return nil, errors.New("FFNet is not finished, use End() to finish it before using it")
	}

	//Call all nodes concurrently
	for i := range n.nodes {
		go n.nodes[i].Activate()
	}

	//Send the data to the starting node
	n.input <- input

	//Wait for the result to be available and return it
	return <-n.output, nil
}

func (n *FFNet) BackPropagate(input *tensor.Tensor) (*tensor.Tensor, error) {
	if !n.finished {
		return nil, errors.New("FFNet is not finished, use End() to finish it before using it")
	}

	//Call all nodes concurrently
	for i := range n.nodes {
		go n.nodes[i].BackPropagate()
	}

	//Send the data to the starting node
	n.output <- input

	//Wait for the result to be available and return it
	return <-n.input, nil
}

func (n *FFNet) AddLayer(layer weight.Layer, parents ...string) error {

	if n.finished {
		return errors.New("FFNet finished, cannot add more layers")
	}

	if n.nodes == nil {
		n.nodes = []*FFNode{}
	}

	if n.nodeByID(layer.ID()) != nil {
		return errors.New("There's already a layer in the FFNet with the id " + layer.ID())
	}

	node := &FFNode{layer: layer}
	node.outputs = []chan *tensor.Tensor{}
	node.inputs = []chan *tensor.Tensor{}

	if len(n.nodes) == 0 {
		//First node
		n.input = make(chan *tensor.Tensor, 1)
		n.startNode = node
		node.inputs = append(node.inputs, n.input)
	} else {
		if len(parents) == 0 {
			return errors.New("No parent especified")
		}
	}

	for _, parentID := range parents {
		parent := n.nodeByID(parentID)
		if parent == nil {
			return errors.New("Could not find parent layer")
		}
		err := n.setParent(node, parent)
		if err != nil {
			return err
		}
	}

	n.nodes = append(n.nodes, node)

	return nil
}

func (n *FFNet) nodeByID(id string) *FFNode {
	for i := 0; i < len(n.nodes); i++ {
		if n.nodes[i].ID() == id {
			return n.nodes[i]
		}
	}

	return nil
}

func (n *FFNet) setParent(node *FFNode, parent *FFNode) error {
	if node == nil {
		return errors.New("FFNode is nil")
	}

	if parent == nil {
		return errors.New("Parent is nil")
	}

	c := make(chan *tensor.Tensor, 1)

	node.parents = append(node.parents, parent)

	node.inputs = append(node.inputs, c)
	parent.outputs = append(parent.outputs, c)

	return nil
}

//End closes the network
func (n *FFNet) End() error {
	last := n.nodes[len(n.nodes)-1]
	n.output = make(chan *tensor.Tensor, 1)
	n.endNode = last

	last.outputs = []chan *tensor.Tensor{n.output}

	n.finished = true

	return nil
}

func (n *FFNet) GetOutputSize() []int {
	return n.endNode.layer.GetOutputSize()
}

func (n *FFNet) GetInputSize() []int {
	return n.startNode.layer.GetInputSize()
}

func (n *FFNet) GetParamGradPointers() ([]*float64, []*float64) {
	params := []*float64{}
	grads := []*float64{}

	for i := 0; i < len(n.nodes); i++ {
		bpLayer, ok := n.nodes[i].layer.(weight.BPLearnerLayer)
		if !ok {
			panic("weight.Layer inside FFNet does not implement weight.BPLearnerLayer interface")
		}

		p, g := bpLayer.GetParamGradPointers()
		params = append(params, p...)
		grads = append(grads, g...)
	}
	return params, grads
}

func (n *FFNet) GetDebugInfo() []*debug.LayerInfo {

	stats := []*debug.LayerInfo{}

	for i := 0; i < len(n.nodes); i++ {
		dLayer, ok := n.nodes[i].layer.(debug.DebugLayer)
		if ok {
			stats = append(stats, dLayer.GetDebugInfo()...)
		}

	}

	return stats
}

func NewSequentialNet(layers ...weight.Layer) (*FFNet, error) {
	net := NewFFNet()

	id := layers[0].ID()
	err := net.AddLayer(layers[0])
	if err != nil {
		return nil, err
	}

	for i := 1; i < len(layers); i++ {
		err := net.AddLayer(layers[i], id)
		if err != nil {
			return nil, err
		}
		id = layers[i].ID()
	}

	net.End()

	return net, nil
}
