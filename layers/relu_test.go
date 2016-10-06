package layers

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMarshalReLU(t *testing.T) {
	assert := assert.New(t)

	l := NewReLULayer(1, 9, 6)

	var b bytes.Buffer

	l.Marshal(&b)

	unmarshaled, err := UnmarshalReLULayer(&b)
	if err != nil {
		assert.FailNow("Error while unmarshaling")
	}

	nl := unmarshaled.(*ReLULayer)

	assert.InDeltaSlice(nl.GetInputSize(), l.GetInputSize(), 1e-6, "Expected same input size")
	assert.InDeltaSlice(nl.GetOutputSize(), l.GetOutputSize(), 1e-6, "Expected same output size")
	assert.InDelta(nl.negativeSlope, l.negativeSlope, 1e-6, "Expected same negative slope")
	assert.EqualValues(nl.id, l.id, "Expected same id")
}
