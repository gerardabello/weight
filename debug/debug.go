package debug

import (
	"image"

	"gitlab.com/gerardabello/weight"
	"gitlab.com/gerardabello/weight/tensor"
)

type DebugLayer interface {
	weight.Layer

	GetDebugInfo() []*LayerInfo
}

type LayerInfo struct {
	ID string

	WeightsImg []image.Image
	BiasImg    []image.Image
	OutImg     []image.Image
	ErrImg     []image.Image

	WeightsStats *tensor.Stats
	BiasStats    *tensor.Stats
	OutStats     *tensor.Stats
	ErrStats     *tensor.Stats

	Other map[string]interface{}
}

type TrainInfo struct {
	Epoch             int
	Epochs            int
	Batch             int
	Batches           int
	Loss              float64
	Accuracy          float64
	ExamplesPerSecond float64
}

type TestInfo struct {
	Epoch    int
	Loss     float64
	Accuracy float64
}

type NetDebugger interface {
	Debug(status <-chan string, layerInfo <-chan []*LayerInfo, trainInfo <-chan *TrainInfo, testInfo <-chan *TestInfo)
}
