package debug

import "fmt"

type CLIDebugger struct {
	showLayerInfo bool
}

func (d *CLIDebugger) Debug(status <-chan string, layerInfo <-chan []*LayerInfo, trainInfo <-chan *TrainInfo, testInfo <-chan *TestInfo) {
	for {
		select {
		case s := <-status:
			fmt.Printf("#   %s\n", s)
		case layerStats := <-layerInfo:
			if d.showLayerInfo {
				println("Layers:")
				for _, stat := range layerStats {
					fmt.Println(stat.OutStats)
				}
			}
		case trainerStats := <-trainInfo:
			fmt.Printf("epoch %5.2f - loss:%-8.4f accuracy:%-8.4f EPS:%-8.1f\n", float64(trainerStats.Epoch)+(float64(trainerStats.Batch)/float64(trainerStats.Batches)), trainerStats.Loss, trainerStats.Accuracy, trainerStats.ExamplesPerSecond)

		case testStats := <-testInfo:
			fmt.Printf("Test Results: \n\t accuracy:%.4f \n\t loss:%.4f \n", testStats.Accuracy, testStats.Loss)
		}
	}
}
