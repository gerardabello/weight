package tensor

import (
	"io"

	"gitlab.com/gerardabello/weight/loaders/utils/idx"
)

func (t *Tensor) Marshal(w io.Writer) error {
	var size []int32
	for _, s := range t.Size {
		size = append(size, int32(s))
	}
	iw := idx.NewWriter(w, idx.Float64DataType, size)

	return iw.WriteFloat64(t.Values)
}

func Unmarshal(r io.Reader) (*Tensor, error) {
	rd, err := idx.NewReader(r)

	if err != nil {
		return nil, err
	}

	t := &Tensor{}
	t.Allocate(rd.Dimensions...)

	err = rd.ReadFloat64(t.Values)

	if err != nil {
		return nil, err
	}

	return t, nil
}
