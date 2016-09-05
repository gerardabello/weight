package tensor

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMarshal(t *testing.T) {
	assert := assert.New(t)

	n := 30
	for i := 0; i < n; i++ {

		t := generateRandomTensor()
		var b bytes.Buffer

		t.Marshal(&b)

		nt, err := Unmarshal(&b)

		assert.NoError(err, "Error when loading tensor should be null")

		assert.InEpsilonSlice(t.Values, nt.Values, 0.0001, "Tensor values should be the same after saving and loading")

	}

}
