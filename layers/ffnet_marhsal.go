package layers

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"io"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/marshaling"
)

func (net *FFNet) getParentsMap() map[string][]string {
	res := map[string][]string{}

	for _, node := range net.nodes {
		parents := []string{}
		for _, p := range node.parents {
			parents = append(parents, p.ID())
		}
		res[node.ID()] = parents
	}

	return res
}

func (net *FFNet) Marshal(writer io.Writer) error {
	tarfile := tar.NewWriter(writer)
	defer tarfile.Close()

	//Save parents
	{
		buf, err := json.Marshal(net.getParentsMap())
		if err != nil {
			return err
		}

		err = writeBytesTar(tarfile, buf, "parents")
		if err != nil {
			return err
		}
	}

	//Save layers
	for i := 0; i < len(net.nodes); i++ {

		var buf bytes.Buffer

		ml, ok := net.nodes[i].layer.(weight.MarshalLayer)
		if !ok {
			panic("weight.Layer inside FFNet does not implement MarshalLayer interface")
		}

		err := ml.Marshal(&buf)
		if err != nil {
			return err
		}

		b := buf.Bytes()

		hdr := &tar.Header{
			Name: ml.ID() + "." + ml.GetName(),
			Mode: 0644,
			Size: int64(len(b)),
		}

		err = tarfile.WriteHeader(hdr)
		if err != nil {
			return err
		}

		_, err = tarfile.Write(b)
		if err != nil {
			return err
		}

	}
	return nil
}

const FFNetName = "ffnet"

func init() {
	marshaling.RegisterFormat(FFNetName, UnmarshalFFNet)
}

func (l *FFNet) GetName() string {
	return FFNetName
}

func UnmarshalFFNet(reader io.Reader) (weight.MarshalLayer, error) {
	tr := tar.NewReader(reader)
	// Iterate through the files in the archive.

	var inputSize []int

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

	return l, nil
}
