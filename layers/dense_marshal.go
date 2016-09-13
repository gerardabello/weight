package layers

import (
	"archive/tar"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/rand"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/marshaling"
	"gitlab.com/gerardabello/weight/tensor"
)


func (l *DenseLayer) Marshal(writer io.Writer) error {
	tarfile := tar.NewWriter(writer)
	defer tarfile.Close()

	//save info
	err := writeInfoTar(
		tarfile,
		&map[string]interface{}{
			"input":  l.GetInputSize(),
			"output": l.GetOutputSize(),
		},
	)
	if err != nil {
		return err
	}

	err = writeTensorTar(tarfile, l.bias, "bias")
	if err != nil {
		return err
	}

	err = writeTensorTar(tarfile, l.weights, "weights")
	if err != nil {
		return err
	}

	return nil

}

const DenseName = "dense"

func init() {
	marshaling.RegisterFormat(DenseName, UnmarshalDenseLayer)
}

func (l *DenseLayer) GetName() string {
	return DenseName
}

func UnmarshalDenseLayer(reader io.Reader) (weight.MarshalLayer, error) {
	tr := tar.NewReader(reader)
	// Iterate through the files in the archive.

	var inputSize []int
	var outputSize []int

	var weights *tensor.Tensor
	var bias *tensor.Tensor

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			// end of tar archive
			break
		}
		if err != nil {
			return nil, err
		}

		switch hdr.Name {
		case "info":
			info := map[string][]int{}
			dec := json.NewDecoder(tr)
			err = dec.Decode(&info)
			if err == io.EOF {
				continue
			}
			if err != nil {
				return nil, err
			}

			inputSize = info["input"]
			outputSize = info["output"]

		case "weights":
			weights, err = tensor.Unmarshal(tr)
			if err != nil {
				return nil, err
			}
		case "bias":
			bias, err = tensor.Unmarshal(tr)
			if err != nil {
				return nil, err

			}
		default:
			return nil, errors.New("Unrecognized file " + hdr.Name)
		}

	}

	l := NewDenseLayer(inputSize, outputSize)

	l.weights = weights
	l.bias = bias

	return l, nil
}
