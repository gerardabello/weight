package imgfolder

import (
	"bufio"
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"strconv"

	"gitlab.com/gerardabello/weight/loaders/utils"
	"gitlab.com/gerardabello/weight/tensor"
)

type img struct {
	path  string
	label int
}

type FolderSet struct {
	imgs []img

	nlabels int
	imgsize [2]int

	mutex *sync.Mutex

	pointer int
}

func (m *FolderSet) GetDataSize() []int {
	return m.imgsize[:]
}

func (m *FolderSet) GetAnswersSize() []int {
	return []int{m.nlabels}
}

func (m *FolderSet) GetSetSize() int {
	return len(m.imgs)
}

func (m *FolderSet) Reset() {
	m.mutex.Lock()
	m.pointer = 0
	m.mutex.Unlock()
}

func (m *FolderSet) GetNextSet() (*tensor.Tensor, *tensor.Tensor, error) {
	m.mutex.Lock()
	im, err := utils.LoadImage(m.imgs[m.pointer].path, utils.RGB)
	if err != nil {
		return nil, nil, err
	}

	if im.Size[0] != m.imgsize[0] || im.Size[1] != m.imgsize[1] {
		return nil, nil, errors.New("Image file does not have expected dimensions")
	}

	lbl := tensor.NewTensor(m.nlabels)
	lbl.SetVal(1.0, m.imgs[m.pointer].label)

	m.pointer++
	m.mutex.Unlock()
	return im, lbl, nil
}

func (m *FolderSet) IsAnswer(out *tensor.Tensor, ans *tensor.Tensor) bool {
	maxOutIndex, _ := out.Max()
	maxAnsIndex, _ := ans.Max()

	return maxOutIndex == maxAnsIndex
}

func (m *FolderSet) Close() {
}

func NewSet(path string) (*FolderSet, error) {
	//New structure

	set := &FolderSet{}

	file, err := os.Open(filepath.Join(path, "info"))
	if err != nil {
		return nil, fmt.Errorf("Error loading info file: %s", err.Error())
	}

	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)

	//read firts line
	{
		scanner.Scan()
		fl := scanner.Text()
		fls := strings.Split(fl, "x")
		set.imgsize[0], err = strconv.Atoi(fls[0])
		set.imgsize[1], err = strconv.Atoi(fls[1])

		if err != nil {
			return nil, fmt.Errorf("Error loading info file. Could not parse image size: %s", err.Error())
		}
	}

	index := 0
	for scanner.Scan() {
		folders := strings.Split(scanner.Text(), "+")
		for _, folder := range folders {
			timg, err := openLabelFolder(filepath.Join(path, folder), index)
			if err != nil {
				return nil, fmt.Errorf("Error loading %s folder: %s", folder, err.Error())
			}
			set.imgs = append(set.imgs, timg...)
		}
		index++
	}

	shuffle(set.imgs)

	set.nlabels = index + 1
	set.pointer = 0
	set.mutex = &sync.Mutex{}

	return set, nil
}

func openLabelFolder(path string, lbl int) ([]img, error) {
	imgs := []img{}

	files, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}

	for _, file := range files {
		if !file.IsDir() {
			imgs = append(imgs, img{label: lbl, path: filepath.Join(path, file.Name())})
		}
	}

	return imgs, nil
}

func shuffle(slc []img) {
	N := len(slc)
	for i := 0; i < N; i++ {
		// choose index uniformly in [i, N-1]
		r := i + rand.Intn(N-i)
		slc[r], slc[i] = slc[i], slc[r]
	}
}
