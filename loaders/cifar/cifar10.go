package cifar

import (
	"path/filepath"
	"reflect"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

func OpenCIFAR10(base string) (*weight.PairSet, error) {
	trainSet, err := NewCIFAR10Set([]string{
		filepath.Join(base + `data_batch_1.bin`),
		filepath.Join(base + `data_batch_2.bin`),
		filepath.Join(base + `data_batch_3.bin`),
		filepath.Join(base + `data_batch_4.bin`),
		filepath.Join(base + `data_batch_5.bin`),
	})
	if err != nil {
		return nil, err
	}

	testSet, err := NewCIFAR10Set([]string{
		filepath.Join(base + `test_batch.bin`),
	})
	if err != nil {
		return nil, err
	}

	return &weight.PairSet{TrainSet: trainSet, TestSet: testSet}, nil
}

type CIFAR10Set struct {
	CIFAR
}

type CIFAR10Row struct {
	Label byte
	Img   [32 * 32 * 3]byte
}

func (m *CIFAR10Set) GetAnswersSize() []int {
	return []int{10}
}

func (m *CIFAR10Set) GetNextSet() (*tensor.Tensor, *tensor.Tensor, error) {

	row := CIFAR10Row{}
	err := m.CIFAR.GetNextRow(&row)

	if err != nil {
		return nil, nil, err
	}

	lbl, img := createCifarData(m.GetAnswersSize(), m.GetDataSize(), row.Label, row.Img)

	return lbl, img, nil
}

func NewCIFAR10Set(paths []string) (*CIFAR10Set, error) {

	//New structure
	set := &CIFAR10Set{}
	err := set.CIFAR.init(paths, reflect.TypeOf(CIFAR10Row{}))

	return set, err
}
