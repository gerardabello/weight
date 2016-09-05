package tensor

func (t *Tensor) Slice(indexes ...int) *Tensor {
	dim := t.GetDims() - len(indexes)

	if dim < 0 {
		panic("Number of indexes is bigger than tensor dimension")
	}

	pos := 0
	siz := 1
	size := 0
	for i := 0; i < t.GetDims(); i++ {
		if i >= dim {
			pos += siz * indexes[len(indexes)-i+dim-1]
		}

		if i == dim {
			size = siz
		}

		siz *= t.Size[i]

	}

	s := t.Size[0:dim]
	v := t.Values[pos : pos+size]

	return &Tensor{Size: s, Values: v}
}
