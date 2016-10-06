package layers

import (
	"archive/tar"
	"encoding/json"
	"errors"
	"io"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/marshaling"
)

func (l *ReLULayer) Marshal(writer io.Writer) error {
	tarfile := tar.NewWriter(writer)
	defer tarfile.Close()

	//save info
	err := writeInfoTar(
		tarfile,
		&map[string]interface{}{
			"input":         l.GetInputSize(),
			"negativeSlope": l.negativeSlope,
			"id":            l.id,
		},
	)
	if err != nil {
		return err
	}

	return nil
}

const ReLUName = "relu"

func init() {
	marshaling.RegisterFormat(ReLUName, UnmarshalReLULayer)
}

func (l *ReLULayer) GetName() string {
	return ReLUName
}

func UnmarshalReLULayer(reader io.Reader) (weight.MarshalLayer, error) {
	tr := tar.NewReader(reader)
	// Iterate through the files in the archive.

	var inputSize []int
	var id string
	var negativeSlope float64

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
			info := map[string]interface{}{}
			dec := json.NewDecoder(tr)
			err := dec.Decode(&info)
			if err == io.EOF {
				continue
			}
			if err != nil {
				return nil, err
			}

			var ok bool
			inputSize, ok = InterfaceArrayToIntArray(info["input"])
			if !ok {
				return nil, errors.New("Unmarshal relu: error parsing info file: could not parse input field")
			}
			id, ok = info["id"].(string)
			if !ok {
				return nil, errors.New("Unmarshal relu: error parsing info file: could not parse id field")
			}
			negativeSlope, ok = info["negativeSlope"].(float64)
			if !ok {
				return nil, errors.New("Unmarshal relu: error parsing info file: could not parse negativeSlope field")
			}

		default:
			return nil, errors.New("Unrecognized file " + hdr.Name)
		}

	}

	l := NewReLULayer(inputSize...)

	l.id = id
	l.negativeSlope = negativeSlope

	return l, nil
}
