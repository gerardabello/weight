package cifar

import (
	"path/filepath"
	"reflect"

	"github.com/gerardabello/weight"
	"github.com/gerardabello/weight/tensor"
)

func OpenCIFAR100(base string) (*weight.PairSet, error) {
	trainSet, err := NewCIFAR100Set([]string{
		filepath.Join(base + `train.bin`),
	})
	if err != nil {
		return nil, err
	}

	testSet, err := NewCIFAR100Set([]string{
		filepath.Join(base + `test.bin`),
	})
	if err != nil {
		return nil, err
	}

	return &weight.PairSet{TrainSet: trainSet, TestSet: testSet}, nil
}

type CIFAR100Set struct {
	CIFAR
}

type CIFAR100Row struct {
	Label    byte
	SubLabel byte
	Img      [32 * 32 * 3]byte
}

func (m *CIFAR100Set) GetAnswersSize() []int {
	return []int{20}
}

func (m *CIFAR100Set) GetNextSet() (*tensor.Tensor, *tensor.Tensor, error) {

	row := CIFAR100Row{}
	err := m.CIFAR.GetNextRow(&row)

	if err != nil {
		return nil, nil, err
	}

	lbl, img := createCifarData(m.GetAnswersSize(), m.GetDataSize(), row.Label, row.Img)

	return lbl, img, nil
}

func NewCIFAR100Set(paths []string) (*CIFAR100Set, error) {

	//New structure
	set := &CIFAR100Set{}
	err := set.CIFAR.init(paths, reflect.TypeOf(CIFAR100Row{}))

	return set, err
}
