package utils

import (
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/gerardabello/weight/tensor"
)

const (
	RGB = iota
	AVERAGE
)

func LoadImage(path string, mode int) (*tensor.Tensor, error) {

	imgfile, err := os.Open(path)

	if err != nil {
		return nil, err
	}
	defer imgfile.Close()

	img, _, err := image.Decode(imgfile)
	if err != nil {
		return nil, err
	}

	t := &tensor.Tensor{}

	switch mode {
	case RGB:
		t.Allocate(img.Bounds().Size().X, img.Bounds().Size().Y, 3)

	case AVERAGE:
		t.Allocate(img.Bounds().Size().X, img.Bounds().Size().Y, 1)
	}

	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			switch mode {
			case RGB:
				t.SetVal(float64(r)/0xffff, x, y, 0)
				t.SetVal(float64(g)/0xffff, x, y, 1)
				t.SetVal(float64(b)/0xffff, x, y, 2)

			case AVERAGE:
				val := float64(r)/0xffff + float64(g)/0xffff + float64(b)/0xffff
				t.SetVal(val/3, x, y, 0)
			}
		}
	}

	return t, nil
}
