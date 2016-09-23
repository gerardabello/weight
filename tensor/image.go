package tensor

import (
	"fmt"
	"image"
	"image/color"
)

func (t *Tensor) Image(dynamicRange float64) (image.Image, error) {
	//TODO optimize this
	var width, height int
	if t.GetDims() == 1 {
		width = t.Size[0]
		height = 1
	} else if t.GetDims() == 2 {
		width = t.Size[0]
		height = t.Size[1]
	} else {
		return nil, fmt.Errorf("Cannot create an image from a tensor with %d dimensions", t.GetDims())
	}

	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {

			var val float64
			if t.GetDims() == 1 {
				val = t.GetVal(x)
			} else {
				val = t.GetVal(x, y)
			}

			var rval, gval float64
			if val < 0 {
				if val < -dynamicRange {
					val = -dynamicRange
				}
				rval = -val / dynamicRange * 255
			} else if val > 0 {
				if val > dynamicRange {
					val = dynamicRange
				}
				gval = val / dynamicRange * 255
			}

			c := color.RGBA{R: uint8(rval), G: uint8(gval), B: 0, A: 255}
			img.SetRGBA(x, y, c)

		}
	}

	return img, nil

}

func (t *Tensor) ImageSlice() ([]image.Image, error) {
	if t.GetDims() == 1 {
		im, err := t.Image(t.StdDev() * 3)
		return []image.Image{im}, err
	}

	if t.GetDims() != 3 {
		return nil, fmt.Errorf("Cannot create an image slice from a tensor with %d dimensions", t.GetDims())
	}

	n := t.Size[2]

	imgs := []image.Image{}

	for i := 0; i < n; i++ {
		st := t.Slice(i)
		im, err := st.Image(t.StdDev() * 3)

		if err != nil {
			return nil, err
		}

		imgs = append(imgs, im)

	}

	return imgs, nil

}
