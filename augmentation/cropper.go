package augmentation

import (
	"errors"
	"math/rand"
	"sync"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

type Cropper struct {
	weight.DataSet
	MaxAmount []int

	mutex sync.Mutex
}

func (s *Cropper) GetDataSize() []int {
	size := s.DataSet.GetDataSize()
	for i := 0; i < len(size); i++ {
		size[i] -= s.MaxAmount[i] * 2
	}

	return size
}

func (s *Cropper) GetNextSet() (*tensor.Tensor, *tensor.Tensor, error) {
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

		amount[i] -= s.MaxAmount[i]
	}

	ndata := tensor.NewTensor(s.GetDataSize()...)

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
