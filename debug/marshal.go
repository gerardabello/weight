package debug

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"

	"gitlab.com/gerardabello/weight/tensor"
)

//Encode images to png/base64 if marshaling to json
func (li *LayerInfo) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		ID         string
		WeightsImg []string
		BiasImg    []string
		OutImg     []string
		ErrImg     []string

		WeightsStats *tensor.Stats
		BiasStats    *tensor.Stats
		OutStats     *tensor.Stats
		ErrStats     *tensor.Stats

		Other map[string]interface{}
	}{
		ID: li.ID,

		WeightsImg: ImageSliceToBase64(li.WeightsImg),
		BiasImg:    ImageSliceToBase64(li.BiasImg),
		OutImg:     ImageSliceToBase64(li.OutImg),
		ErrImg:     ImageSliceToBase64(li.ErrImg),

		WeightsStats: li.WeightsStats,
		BiasStats:    li.BiasStats,
		OutStats:     li.OutStats,
		ErrStats:     li.ErrStats,

		Other: li.Other,
	})
}

func ImageSliceToBase64(imgs []image.Image) []string {
	simgs := make([]string, len(imgs))

	for i := 0; i < len(imgs); i++ {
		buf := bytes.Buffer{}
		png.Encode(&buf, imgs[i])

		// convert the buffer bytes to base64 string - use buf.Bytes() for new image
		simgs[i] = base64.StdEncoding.EncodeToString(buf.Bytes())
	}

	return simgs
}
