package layers

import (
	"archive/tar"
	"bytes"
	"encoding/json"

	"gitlab.com/gerardabello/weight/tensor"
)

func writeBytesTar(tarfile *tar.Writer, buf []byte, name string) error {

	hdr := &tar.Header{
		Name: name,
		Mode: 0644,
		Size: int64(len(buf)),
	}

	err := tarfile.WriteHeader(hdr)
	if err != nil {
		return err
	}

	_, err = tarfile.Write(buf)
	if err != nil {
		return err
	}

	return nil
}

func writeInfoTar(tarfile *tar.Writer, info *map[string]interface{}) error {
	buf, err := json.Marshal(info)
	if err != nil {
		return err
	}

	return writeBytesTar(tarfile, buf, "info")
}

func writeTensorTar(tarfile *tar.Writer, t *tensor.Tensor, name string) error {
	buf := new(bytes.Buffer)
	t.Marshal(buf)

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
