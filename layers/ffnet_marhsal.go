package layers

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"

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

func (*FFNet) GetName() string {
	return FFNetName
}

func UnmarshalFFNet(reader io.Reader) (weight.MarshalLayer, error) {
	tr := tar.NewReader(reader)
	// Iterate through the files in the archive.

	layers := map[string]weight.MarshalLayer{}
	parents := map[string][]string{}

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			// end of tar archive
			break
		}
		if err != nil {
			return nil, err
		}

		if hdr.Name == "parents" {
			dec := json.NewDecoder(tr)
			err := dec.Decode(&parents)

			if err != nil {
				return nil, err
			}
		} else {

			str := strings.Split(hdr.Name, ".")
			id := str[0]
			layerName := str[1]
			layers[id], err = marshaling.Unmarshal(layerName, tr)
			if err != nil {
				return nil, err
			}
		}
	}

	net := NewFFNet()

	count := 0
	for {

		for k, v := range parents {
			err := net.AddLayer(layers[k], v...)
			if err == nil {
				delete(parents, k)
			}

		}

		if len(parents) == 0 {
			break
		}

		if count > 1000 {
			return nil, errors.New("Could not resolve graph")
		}

		count++
	}

	net.End()

	return net, nil
}
