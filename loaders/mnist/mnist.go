package mnist

import (
	"io"
	"net/http"
	"os"
	"path/filepath"

	"gitlab.com/gerardabello/weight"
)

func Open(base string) (*weight.PairSet, error) {
	trainSet, err := NewSet(
		filepath.Join(base, `train-images-idx3-ubyte.gz`),
		filepath.Join(base, `train-labels-idx1-ubyte.gz`),
	)
	if err != nil {
		return nil, err
	}

	testSet, err := NewSet(
		filepath.Join(base, `t10k-images-idx3-ubyte.gz`),
		filepath.Join(base, `t10k-labels-idx1-ubyte.gz`),
	)
	if err != nil {
		return nil, err
	}

	return &weight.PairSet{TrainSet: trainSet, TestSet: testSet}, nil
}

func downloadFile(filepath string, url string) (err error) {

	// New the file
	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	// Get the data
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Writer the body to file
	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return err
	}

	return nil
}
