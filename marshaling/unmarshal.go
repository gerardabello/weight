package marshaling

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"

	"gitlab.com/gerardabello/weight"
)

// A format holds a layer's name, and how to decode it.
type format struct {
	name      string
	unmarshal func(io.Reader) (weight.MarshalLayer, error)
}

// Formats is the list of registered formats.
var formats []format

// RegisterFormat registers an format for use by Unmarshal.
func RegisterFormat(name string, unmarshal func(io.Reader) (weight.MarshalLayer, error)) {
	formats = append(formats, format{strings.ToLower(name), unmarshal})
}

// Unmarshal returns a layer
func Unmarshal(name string, reader io.Reader) (weight.MarshalLayer, error) {
	lname := strings.ToLower(name)
	for _, f := range formats {
		if f.name == lname {
			return f.unmarshal(reader)
		}
	}

	return nil, errors.New("Unknown layer format")
}

func UnmarshalFromFile(file string) (weight.MarshalLayer, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	ext := filepath.Ext(file)

	return Unmarshal(ext, f)
}
