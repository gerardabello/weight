package augmentation

import (
	"math/rand"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

type Scaler struct {
	weight.DataSet
	StDev float64
	Mean  float64
}

func (s *Scaler) GetNextSet() (*tensor.Tensor, *tensor.Tensor, error) {
	data, ans, err := s.DataSet.GetNextSet()
	if err != nil {
		return nil, nil, err
	}

	scale := rand.NormFloat64()*s.StDev + s.Mean
	data.Mul(scale)

	return data, ans, nil
}
