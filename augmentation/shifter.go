package augmentation

import (
	"errors"
	"math/rand"
	"sync"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

//ShiftMethod defines the method used to update parameters in each layer
type ShiftMethod int

const (
	//Copy just copies the moved pixels, leaving the old ones where no new pixel is assigned
	Copy ShiftMethod = iota

	//Zeros sets zeros to places where a pixel is not assigned after the shift
	Zeros
)

type Shifter struct {
	weight.DataSet
	MaxAmount []int
	Method    ShiftMethod

	mutex sync.Mutex
}

func (s *Shifter) GetNextSet() (*tensor.Tensor, *tensor.Tensor, error) {
	data, ans, err := s.DataSet.GetNextSet()
	if err != nil {
		return nil, nil, err
	}

	if len(data.Size) != len(s.MaxAmount) {
		return nil, nil, errors.New("Shifter.MaxAmount does not have the same length as input.Size")
	}

	s.mutex.Lock()
	defer s.mutex.Unlock()

	amount := make([]int, len(data.Size))
	for i := 0; i < len(data.Size); i++ {
		if s.MaxAmount[i] > 0 {
			amount[i] = rand.Intn(s.MaxAmount[i])
			if rand.NormFloat64() > 0 {
				amount[i] = -amount[i]
			}
		}
	}

	var ndata *tensor.Tensor
	if s.Method == Copy {
		ndata = data.Copy()
	} else {
		ndata = tensor.NewTensor(data.Size...)
	}

	n := data.GetNumberOfValues()
	for i := 0; i < n; i++ {
		p := data.FlatToDim(i)
		for j := 0; j < len(amount); j++ {
			p[j] += amount[j]
		}
		if ndata.InBounds(p...) {
			ndata.SetVal(data.Values[i], p...)
		}
	}

	return ndata, ans, nil
}
