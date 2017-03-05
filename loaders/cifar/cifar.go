package cifar

import (
	"fmt"
	"os"
	"reflect"
	"sync"

	"github.com/gerardabello/weight/tensor"
)

type CIFAR struct {
	batches      []*os.File
	batchRows    []int
	currentBatch int

	rowType reflect.Type

	mutex *sync.Mutex
}

func (m *CIFAR) GetDataSize() []int {
	return []int{32, 32, 3}
}

func (m *CIFAR) GetSetSize() int {
	sum := 0
	for i := 0; i < len(m.batchRows); i++ {
		sum += m.batchRows[i]
	}
	return sum
}

func (m *CIFAR) Reset() {
	m.mutex.Lock()
	//Set all files to the begining
	for i := range m.batches {
		m.batches[i].Seek(0, 0)
	}

	//set the first batch as the current one
	m.currentBatch = 0
	m.mutex.Unlock()
}

func (m *CIFAR) Close() {
	for i := range m.batches {
		m.batches[i].Close()
	}
}

func (m *CIFAR) IsAnswer(out *tensor.Tensor, ans *tensor.Tensor) bool {
	maxOutIndex, _ := out.Max()
	maxAnsIndex, _ := ans.Max()

	return maxOutIndex == maxAnsIndex
}

func (m *CIFAR) init(paths []string, rowType reflect.Type) error {

	m.mutex = &sync.Mutex{}

	m.batches = make([]*os.File, len(paths))
	m.batchRows = make([]int, len(paths))

	for i, path := range paths {
		var err error
		m.batches[i], err = os.Open(path)

		stat, err := m.batches[i].Stat()
		if err != nil {
			return fmt.Errorf("Could not read file %s: %s", path, err.Error())
		}
		size := stat.Size()

		m.batchRows[i] = int(size) / int(rowType.Size())
		m.rowType = rowType

		if err != nil {
			return err
		}
	}

	return nil
}

func createCifarData(answerSize []int, imageSize []int, label byte, image [3072]byte) (*tensor.Tensor, *tensor.Tensor) {

	img := &tensor.Tensor{}
	lbl := &tensor.Tensor{}

	lbl.Allocate(answerSize...)
	//Set the value in the index 'label' to 1 (the rest default to 0)
	lbl.SetVal(1.0, int(label))

	img.Size = imageSize
	img.Values = make([]float64, len(image)) //32*32*3

	for i := range img.Values {
		img.Values[i] = float64(image[i]) / 255.0
	}

	return img, lbl
}
