package mnist

import (
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"sync"

	"gitlab.com/gerardabello/weight/loaders/utils/idx"
	"gitlab.com/gerardabello/weight/tensor"
)

type MNISTSet struct {
	imgs   []*tensor.Tensor
	labels []*tensor.Tensor

	mutex *sync.Mutex

	readersToClose []io.ReadCloser

	pointer int
}

func (m *MNISTSet) GetDataSize() []int {
	return m.imgs[0].Size
}

func (m *MNISTSet) GetAnswersSize() []int {
	return m.labels[0].Size
}

func (m *MNISTSet) GetSetSize() int {
	return len(m.imgs)
}

func (m *MNISTSet) Reset() {
	m.mutex.Lock()
	m.pointer = 0
	m.mutex.Unlock()
}

func (m *MNISTSet) GetNextSet() (*tensor.Tensor, *tensor.Tensor, error) {
	m.mutex.Lock()
	m.pointer++
	im, lbl, err := m.imgs[m.pointer-1], m.labels[m.pointer-1], error(nil)
	m.mutex.Unlock()
	return im, lbl, err
}

func (m *MNISTSet) IsAnswer(out *tensor.Tensor, ans *tensor.Tensor) bool {
	maxOutIndex, _ := out.Max()
	maxAnsIndex, _ := ans.Max()

	return maxOutIndex == maxAnsIndex
}

func (m *MNISTSet) Close() {
	//Mnist is all stored in memory so no closing is needed
}

func NewSet(imgsPath, labelsPath string) (*MNISTSet, error) {
	//New structure
	set := &MNISTSet{}

	//After reading all, close all the readers
	defer func() {
		for i := range set.readersToClose {
			err := set.readersToClose[i].Close()
			if err != nil {
				panic(err)
			}
		}
	}()

	imgs, err := set.openFile(imgsPath)
	if err != nil {
		return nil, err
	}

	labels, err := set.openFile(labelsPath)
	if err != nil {
		return nil, err
	}

	//Checks
	if len(labels.Dimensions) != 1 {
		return nil, fmt.Errorf("Labels should have 1 dimension")
	}

	if len(imgs.Dimensions) != 3 {
		return nil, fmt.Errorf("Images should have 3 dimensions")
	}

	if imgs.Dimensions[0] != labels.Dimensions[0] {
		return nil, fmt.Errorf("There should be the same number of train images and labels")
	}

	set.imgs = make([]*tensor.Tensor, imgs.Dimensions[0])
	set.labels = make([]*tensor.Tensor, labels.Dimensions[0])

	for i := 0; i < imgs.Dimensions[0]; i++ {
		set.labels[i] = &tensor.Tensor{}
		label := make([]uint8, 1)
		err := labels.ReadUint8(label)
		if err != nil {
			return nil, err
		}
		set.labels[i].Allocate(10)

		//Transform the label (0 to 9) into an array of 0s and a '1'
		set.labels[i].SetVal(1, int(label[0]))

		img := make([]uint8, imgs.Dimensions[1]*imgs.Dimensions[2])
		err = imgs.ReadUint8(img)
		if err != nil {
			return nil, err
		}
		imgFloat := make([]float64, len(img))
		for i := range img {
			imgFloat[i] = float64(img[i]) / 255.0
		}
		set.imgs[i] = &tensor.Tensor{Size: imgs.Dimensions[1:], Values: imgFloat}

	}

	set.mutex = &sync.Mutex{}

	return set, nil
}

func (m *MNISTSet) openFile(filename string) (*idx.Reader, error) {
	fi, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	m.readersToClose = append(m.readersToClose, fi)

	fz, err := gzip.NewReader(fi)
	if err != nil {
		return nil, err
	}
	m.readersToClose = append(m.readersToClose, fz)

	ret, err := idx.NewReader(fz)
	if err != nil {
		return nil, err
	}

	return ret, nil
}
