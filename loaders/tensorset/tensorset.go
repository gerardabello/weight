package tensorset

import (
	"fmt"
	"sync"

	"gitlab.com/gerardabello/weight/tensor"
)

//TensorSet is just a simple Dataset created from two tensor arrays: the data and the expected answers
type TensorSet struct {
	data []*tensor.Tensor
	ans  []*tensor.Tensor

	mutex *sync.Mutex

	pointer int
}

func (m *TensorSet) GetDataSize() []int {
	return m.data[0].Size
}

func (m *TensorSet) GetAnswersSize() []int {
	return m.ans[0].Size
}

func (m *TensorSet) GetSetSize() int {
	return len(m.data)
}

func (m *TensorSet) Reset() {
	m.mutex.Lock()
	m.pointer = 0
	m.mutex.Unlock()
}

func (m *TensorSet) GetNextSet() (*tensor.Tensor, *tensor.Tensor, error) {
	m.mutex.Lock()
	if m.pointer >= len(m.data) {
		return nil, nil, fmt.Errorf("No next set. %d >= %d", m.pointer, len(m.data))
	}

	data := m.data[m.pointer]
	ans := m.ans[m.pointer]

	m.pointer++
	m.mutex.Unlock()
	return data, ans, nil
}

func (m *TensorSet) IsAnswer(out *tensor.Tensor, ans *tensor.Tensor) bool {
	return false
}

func (m *TensorSet) Close() {
}

func NewTensorSet(data, ans []*tensor.Tensor) *TensorSet {
	set := &TensorSet{}

	set.mutex = &sync.Mutex{}

	set.data = data
	set.ans = ans

	set.Reset()

	return set
}
