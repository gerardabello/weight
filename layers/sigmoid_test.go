package layers

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMarshalSigmoid(t *testing.T) {
	assert := assert.New(t)

	l := NewSigmoidLayer(1, 9, 6)

	var b bytes.Buffer

	l.Marshal(&b)

	unmarshaled, err := UnmarshalSigmoidLayer(&b)
	if err != nil {
		assert.FailNow("Error while unmarshaling: " + err.Error())
	}

	nl := unmarshaled.(*SigmoidLayer)

	assert.InDeltaSlice(nl.GetInputSize(), l.GetInputSize(), 1e-6, "Expected same input size")
	assert.InDeltaSlice(nl.GetOutputSize(), l.GetOutputSize(), 1e-6, "Expected same output size")

	assert.EqualValues(nl.id, l.id, "Expected same id")
}
