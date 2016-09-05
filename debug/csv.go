package debug

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
)

type CSVDebugger struct {
}

func (d *CSVDebugger) Debug(status <-chan string, layerInfo <-chan []*LayerInfo, trainInfo <-chan *TrainInfo, testInfo <-chan *TestInfo) {

	trainFile, err := os.Create("train.csv")
	if err != nil {
		log.Fatalln("error creating train.cvs file:", err)
	}
	defer trainFile.Close()
	trainWriter := csv.NewWriter(trainFile)

	testFile, err := os.Create("test.csv")
	if err != nil {
		log.Fatalln("error creating test.cvs file:", err)
	}
	defer testFile.Close()
	testWriter := csv.NewWriter(testFile)

	for {
		select {
		case trainerStats := <-trainInfo:
			record := []string{
				fmt.Sprint(float64(trainerStats.Epoch) + (float64(trainerStats.Batch) / float64(trainerStats.Batches))),
				fmt.Sprint(trainerStats.Loss),
				fmt.Sprint(trainerStats.Accuracy),
			}
			if err := trainWriter.Write(record); err != nil {
				log.Fatalln("error writing record to csv:", err)
			}
			trainWriter.Flush()

		case testStats := <-testInfo:
			record := []string{
				fmt.Sprint(testStats.Epoch),
				fmt.Sprint(testStats.Accuracy),
			}

			if err := testWriter.Write(record); err != nil {
				log.Fatalln("error writing record to csv:", err)
			}
			testWriter.Flush()
		}
	}
}
