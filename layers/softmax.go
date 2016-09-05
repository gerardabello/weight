package layers

import (
	"archive/tar"
	"fmt"
	"io"
	"math"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

type SoftmaxLayer struct {
	BaseLayer
}

func NewSoftmaxLayer(size ...int) *SoftmaxLayer {
	layer := &SoftmaxLayer{}
	layer.BaseLayer.Init(size, size)
	return layer
}

func (l *SoftmaxLayer) CreateSlave() weight.Layer {
	nl := NewSoftmaxLayer(l.GetInputSize()...)
	nl.id = l.ID()
	return nl
}

func (l *SoftmaxLayer) Activate(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	err := l.BaseLayer.Activate(input)
	if err != nil {
		return nil, err
	}

	l.output.Values = SoftMaxLog(input.Values)

	l.mutex.Unlock()
	return &l.output, nil
}

func softMax(values []float64) []float64 {
	ret := make([]float64, len(values))

	denom := 0.0
	for i, v := range values {
		ret[i] = math.Exp(v)
		denom += ret[i]
	}

	if math.IsInf(denom, 0) {

		fmt.Printf("input:%v\n", values)
		fmt.Printf("denom:%v\n", denom)
		panic("Softmax inf!")

	}

	for i := range ret {
		ret[i] /= denom
		if ret[i] == 0 {
			//So we dont get division by 0
			ret[i] = 1e-150
		}
		if math.IsNaN(ret[i]) {
			fmt.Printf("input:%v\n", values)
			fmt.Printf("denom:%v\n", denom)
			panic("Softmax NaN!")
		}
	}

	return ret
}

func SoftMaxLog(values []float64) []float64 {

	var a float64 = math.Inf(-1)
	for i := 0; i < len(values); i++ {
		if values[i] > a {
			a = values[i]
		}
	}

	Z := 0.0
	for i := 0; i < len(values); i++ {
		Z += math.Exp(values[i] - a)
	}

	ps := make([]float64, len(values))
	for i := 0; i < len(values); i++ {
		ps[i] = math.Exp(values[i]-a) / Z
	}

	return ps
}

func (l *SoftmaxLayer) BackPropagate(err *tensor.Tensor) (*tensor.Tensor, error) {
	l.mutex.Lock()
	e := l.BaseLayer.BackPropagate(err)
	if e != nil {
		return nil, e
	}

	errs := err.Values
	outs := l.output.Values

	for i := range l.propagation.Values {
		for j := range l.propagation.Values {
			if i == j {
				l.propagation.Values[i] += outs[i] * (1 - outs[i]) * errs[j]
			} else {
				l.propagation.Values[i] += -outs[i] * outs[j] * errs[j]
			}
		}
	}

	l.mutex.Unlock()
	return &l.propagation, nil
}

func (l *SoftmaxLayer) Marshal(writer io.Writer) error {
	tarfile := tar.NewWriter(writer)
	defer tarfile.Close()

	//save info
	err := writeInfoTar(
		tarfile,
		&map[string]interface{}{
			"input": l.GetInputSize(),
		},
	)
	if err != nil {
		return err
	}

	return nil
}
