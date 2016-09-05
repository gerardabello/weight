package marshaling

import (
	"os"

	"gitlab.com/gerardabello/weight"
)

func MarshalToFile(layer weight.MarshalLayer, path string) error {

	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE, 0777)
	if err != nil {
		return err
	}
	defer f.Close()

	layer.Marshal(f)

	return nil
}
