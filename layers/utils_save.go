package layers

import (
	"archive/tar"
	"bytes"
	"encoding/json"

	"gitlab.com/gerardabello/weight/tensor"
)

func writeInfoTar(tarfile *tar.Writer, info *map[string]interface{}) error {
	buf, err := json.Marshal(info)
	if err != nil {
		return err
	}

	hdrBias := &tar.Header{
		Name: "info",
		Mode: 0600,
		Size: int64(len(buf)),
	}

	err = tarfile.WriteHeader(hdrBias)
	if err != nil {
		return err
	}

	_, err = tarfile.Write(buf)
	if err != nil {
		return err
	}

	return nil
}

func writeTensorTar(tarfile *tar.Writer, t *tensor.Tensor, name string) error {
	buf := new(bytes.Buffer)
	t.Marshal(buf)

	buf.Len()

	hdrBias := &tar.Header{
		Name: name,
		Mode: 0600,
		Size: int64(buf.Len()),
	}

	err := tarfile.WriteHeader(hdrBias)
	if err != nil {
		return err
	}

	_, err = tarfile.Write(buf.Bytes())
	if err != nil {
		return err
	}

	return nil
}
