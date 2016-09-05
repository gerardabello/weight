package tensor

import (
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestImage(t *testing.T) {
	assert := assert.New(t)
	tt := Tensor{
		Values: []float64{
			0, 0, -2,
			2, 1, -1,
			0, 2, 0},
		Size: []int{3, 3},
	}

	img, err := tt.Image(2)

	assert.NoError(err, "Creating an image should return no error")

	assert.EqualValues(tt.Size[0], img.Bounds().Dx(), "Width of tensor and image should be equal")
	assert.EqualValues(tt.Size[1], img.Bounds().Dy(), "Height of tensor and image should be equal")

	assert.EqualValues(color.RGBA{0, 127, 0, 255}, img.At(1, 1), "Expected value")
	assert.EqualValues(color.RGBA{127, 0, 0, 255}, img.At(2, 1), "Expected value")
	assert.EqualValues(color.RGBA{0, 0, 0, 255}, img.At(1, 0), "Expected value")
	assert.EqualValues(color.RGBA{0, 255, 0, 255}, img.At(1, 2), "Expected value")
	assert.EqualValues(color.RGBA{255, 0, 0, 255}, img.At(2, 0), "Expected value")

}

func TestImageSaturated(t *testing.T) {
	assert := assert.New(t)
	tt := Tensor{
		Values: []float64{
			0, 0, -0.25,
			1, 0.25, -0.5,
			0, 1, 0},
		Size: []int{3, 3},
	}

	img, err := tt.Image(0.5)

	assert.NoError(err, "Creating an image should return no error")

	assert.EqualValues(tt.Size[0], img.Bounds().Dx(), "Width of tensor and image should be equal")
	assert.EqualValues(tt.Size[1], img.Bounds().Dy(), "Height of tensor and image should be equal")

	assert.EqualValues(color.RGBA{0, 127, 0, 255}, img.At(1, 1), "Expected value")
	assert.EqualValues(color.RGBA{255, 0, 0, 255}, img.At(2, 1), "Expected value")
	assert.EqualValues(color.RGBA{0, 0, 0, 255}, img.At(1, 0), "Expected value")
	assert.EqualValues(color.RGBA{0, 255, 0, 255}, img.At(1, 2), "Expected value")
	assert.EqualValues(color.RGBA{127, 0, 0, 255}, img.At(2, 0), "Expected value")

}
