package layers

import (
	"archive/tar"
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/marshaling"
)

func (l *SigmoidLayer) Marshal(writer io.Writer) error {
	tarfile := tar.NewWriter(writer)
	defer tarfile.Close()

	//save info
	err := writeInfoTar(
		tarfile,
		&map[string]interface{}{
			"input": l.GetInputSize(),
		},
	)
	if err != nil {
		return err
	}

	err = writeBytesTar(tarfile, []byte(l.id), "id")
	if err != nil {
		return err
	}

	return nil
}

const SigmoidName = "sigmoid"

func init() {
	marshaling.RegisterFormat(SigmoidName, UnmarshalSigmoidLayer)
}

func (l *SigmoidLayer) GetName() string {
	return SigmoidName
}

func UnmarshalSigmoidLayer(reader io.Reader) (weight.MarshalLayer, error) {
	tr := tar.NewReader(reader)
	// Iterate through the files in the archive.

	var inputSize []int
	var id string

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

		case "id":
			buf, err := ioutil.ReadAll(tr)
			if err != nil {
				return nil, errors.New("id file read: " + err.Error())
			}
			id = string(buf)

		case "info":
			info := map[string][]int{}
			dec := json.NewDecoder(tr)
			err := dec.Decode(&info)
			if err == io.EOF {
				continue
			}
			if err != nil {
				return nil, err
			}

			inputSize = info["input"]

		default:
			return nil, errors.New("Unrecognized file " + hdr.Name)
		}

	}

	l := NewSigmoidLayer(inputSize...)

	l.id = id

	return l, nil
}
