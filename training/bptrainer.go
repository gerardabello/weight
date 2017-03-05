package training

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/gerardabello/weight"
	"github.com/gerardabello/weight/debug"
)

//BPTrainer trains a network given a train set and a learning configuration. It also returns debug information to monitor the process
type BPTrainer struct {
	config       LearningConfig
	data         *weight.PairSet
	net          weight.BPLearnerLayer
	costFunction weight.BPCostFunc

	debugger debug.NetDebugger

	numRoutines int

	params []*float64
	grads  [][]*float64

	//Temp arrays to store values for momentum, adagrad, etc.
	arr1 []float64
	arr2 []float64

	//parameters
	eps   float64
	ro    float64
	beta1 float64
	beta2 float64
}

//NewBPTrainer creates a new BPTrainer
func NewBPTrainer(config LearningConfig, data *weight.PairSet, net weight.BPLearnerLayer, costFunction weight.BPCostFunc) *BPTrainer {
	t := BPTrainer{}
	t.net = net
	t.config = config
	t.data = data
	t.costFunction = costFunction

	p, _ := t.net.GetParamGradPointers()
	t.arr1 = make([]float64, len(p))
	t.arr2 = make([]float64, len(p))

	//TODO include this in learning params
	t.eps = 1e-8
	t.ro = 0.95
	t.beta1 = 0.9   // used in adam
	t.beta2 = 0.999 // used in adam

	t.numRoutines = 4

	return &t
}

//SetDebugger sets the debugger to use during the train process
func (t *BPTrainer) SetDebugger(debugger debug.NetDebugger) {
	t.debugger = debugger
}

func (t *BPTrainer) updateParams(percent float64) {
	k := int(percent * float64(t.config.BatchSize) * float64(t.config.Epochs))

	//smooth transition between start and end
	learningRate := (t.config.LearningRateEnd-t.config.LearningRateStart)*(-math.Pow(2, -10*(percent))+1) + t.config.LearningRateStart

	for p := 0; p < len(t.params); p++ {
		grad := 0.0
		//Calculate mean gradient between each goroutine
		for g := 0; g < t.numRoutines; g++ {
			grad += *t.grads[g][p]

			//Individual gradients are set to zero for the next batch

			*(t.grads[g][p]) = 0
		}

		//calculate gradient of weight decay
		l2grad := t.config.WeightDecay * (*t.params[p])

		//final gradient calculation
		grad = (grad + l2grad) / float64(t.config.BatchSize)

		switch t.config.Method {
		case Momentum:
			dx := -grad*learningRate + t.arr1[p]
			(*t.params[p]) += dx
			t.arr1[p] = dx * t.config.Momentum
		case AdaDelta:
			t.arr1[p] = t.ro*t.arr1[p] + (1-t.ro)*grad*grad
			dx := -math.Sqrt((t.arr2[p]+t.eps)/(t.arr1[p]+t.eps)) * grad
			t.arr2[p] = t.ro*t.arr2[p] + (1-t.ro)*dx*dx // yes, arr2 lags behind arr1 by 1.
			(*t.params[p]) += dx
		case Adam:
			t.arr1[p] = t.arr1[p]*t.beta1 + (1-t.beta1)*grad                // update biased first moment estimate
			t.arr2[p] = t.arr2[p]*t.beta2 + (1-t.beta2)*grad*grad           // update biased second moment estimate
			var biasCorr1 = t.arr1[p] * (1 - math.Pow(t.beta1, float64(k))) // correct bias first moment estimate
			var biasCorr2 = t.arr2[p] * (1 - math.Pow(t.beta2, float64(k))) // correct bias second moment estimate
			var dx = -learningRate * biasCorr1 / (math.Sqrt(biasCorr2) + t.eps)
			(*t.params[p]) += dx
		}

	}

}

//SetNumGoroutines sets the number of parallel goroutines that will train a part of each batch independently
func (t *BPTrainer) SetNumGoroutines(nt int) error {
	if nt <= 0 {
		return errors.New("Cannot have 0 or less threads")
	}
	t.numRoutines = nt
	return nil
}

//Train tries to perform gradient descent using backpropagation
func (t *BPTrainer) Train() error {
	//Everything inline to increase performance

	if t.config.BatchSize == 0 {
		return errors.New("Batch size cannot be 0")
	}

	if t.data.TrainSet.GetSetSize()%t.config.BatchSize != 0 {
		fmt.Printf("Warning: set size (%d) is not divisible by batch size (%d). The remainding data will not be used (last %d elements). \n", t.data.TrainSet.GetSetSize(), t.config.BatchSize, t.data.TrainSet.GetSetSize()%t.config.BatchSize)
	}

	if t.config.BatchSize%t.numRoutines != 0 {
		return fmt.Errorf("Batch size (%d) should be a multiple of the number of goroutines (%d). By default this trainer uses 4 goroutines. If you want to use a fixed value use SetNumGoroutines(int)", t.config.BatchSize, t.numRoutines)
	}

	var layerInfo chan []*debug.LayerInfo
	var trainInfo chan *debug.TrainInfo
	var testInfo chan *debug.TestInfo
	var status chan string

	if t.debugger != nil {
		layerInfo = make(chan []*debug.LayerInfo, 1)
		trainInfo = make(chan *debug.TrainInfo, 1)
		testInfo = make(chan *debug.TestInfo, 1)

		//The status can change quickly (for example if multiple errors are found), so give it a bigger buffer
		status = make(chan string, 10)

		go t.debugger.Debug(status, layerInfo, trainInfo, testInfo)
	}

	layers := []weight.BPLearnerLayer{t.net}
	costFuncs := []weight.BPCostFunc{t.costFunction}

	//New an array of pointers to parameters and their gradients in all layers
	var pg1 []*float64
	t.params, pg1 = t.net.GetParamGradPointers()
	t.grads = [][]*float64{pg1}

	//If we use more than 1 gorotuine, create slave layers to run in parallel
	if t.numRoutines > 1 {
		enslaver, ok := t.net.(weight.EnslaverLayer)
		if !ok {
			return fmt.Errorf("Trainer is configured to use %d goroutines. To use more than one the supplied layer must implement EnslaverLayer interface, but it does not.", t.numRoutines)
		}
		for i := 1; i < t.numRoutines; i++ {
			layers = append(layers, enslaver.CreateSlave().(weight.BPLearnerLayer))
			costFuncs = append(costFuncs, t.costFunction.CreateSlave())
			_, pg := t.net.GetParamGradPointers()
			t.grads = append(t.grads, pg)
		}

		if len(status) < cap(status) {
			status <- fmt.Sprintf("Starting training with %d routines", t.numRoutines)
		}
	}

	//Number of batches
	nbatch := t.data.TrainSet.GetSetSize() / t.config.BatchSize

	var wg sync.WaitGroup

	//temp variables to store accuracy, time, loss, etc.
	accMutex := &sync.Mutex{}
	accCost := 0.0
	accAcTime := 0.0
	accBpTime := 0.0
	nCorrect := 0

	clog := 0
	cBatch := 0

	tt := time.Now()

	for n := 0; n < t.config.Epochs; n++ {
		if len(status) < cap(status) {
			status <- fmt.Sprintf("Starting training of epoch %d", n)
		}
		for i := 0; i < nbatch; i++ {

			//Calculate the number of training items each routine is gonna compute
			sbs := t.config.BatchSize / t.numRoutines

			for g := 0; g < t.numRoutines; g++ {
				wg.Add(1)
				go func(goroutineIndex, subBatchSize int) {
					// Decrement the counter when the goroutine completes.
					defer wg.Done()

					for j := 0; j < subBatchSize; j++ {

						//Get next data and answer
						input, ans, err := t.data.TrainSet.GetNextSet()
						if err != nil {
							panic(err.Error())
						}

						tm := time.Now()

						//Activate the Sequential
						out, err := layers[goroutineIndex].Activate(input)
						if err != nil {
							panic(err.Error())
						}

						//Calculate cost function of classification
						cost := costFuncs[goroutineIndex].Cost(out, ans)

						if t.debugger != nil {
							accMutex.Lock()
							accCost += cost
							accAcTime += time.Since(tm).Seconds()
							if t.data.TrainSet.IsAnswer(out, ans) {
								nCorrect++
							}
							accMutex.Unlock()
						}

						tm = time.Now()
						//Backpropagate = calculate gradients for all params. (we have pointers to all of them in t.grads)
						_, err = layers[goroutineIndex].BackPropagate(costFuncs[goroutineIndex].BackPropagate())

						if err != nil {
							panic(err.Error())
						}

						if t.debugger != nil {
							accMutex.Lock()
							accBpTime += time.Since(tm).Seconds()
							accMutex.Unlock()
						}

					}

				}(g, sbs)
			}

			//Syncronization
			wg.Wait()

			if t.debugger != nil {
				cBatch++
				clog += t.config.BatchSize

				//Print debug info every 2 seconds or in the last batch. Running this code at the end is important as it resets the accumulator variables needed to print debug info.
				if time.Since(tt).Seconds() > 2 || i == nbatch-1 {

					//Only try to send if channel is not full. If we drop some messages we dont care
					if len(layerInfo) < cap(layerInfo) {
						layerInfo <- t.net.(debug.DebugLayer).GetDebugInfo()
					}
					if len(trainInfo) < cap(trainInfo) {
						trainInfo <- &debug.TrainInfo{
							Epoch:             n,
							Epochs:            t.config.Epochs,
							Batch:             i,
							Batches:           nbatch,
							Loss:              accCost / float64(clog),
							Accuracy:          float64(nCorrect) / float64(clog),
							ExamplesPerSecond: float64(clog) / (time.Since(tt).Seconds()),
							//"activationTime":      accAcTime / float64(clog),
							//"backpropagationTime": accBpTime / float64(clog),
							//"batchTime":           time.Since(tt).Seconds() / float64(cBatch),
						}
					}

					//Debug code for performance
					//fmt.Printf("Act: %.2fms   Bp:%.2fms\n", 1000*accAcTime/float64(clog), 1000*accBpTime/float64(clog))

					accCost = 0.0
					accAcTime = 0.0
					accBpTime = 0.0
					nCorrect = 0

					clog = 0
					cBatch = 0

					tt = time.Now()

				}
			}

			t.updateParams(float64(nbatch*n+i) / float64(nbatch*t.config.Epochs))
		}

		if t.debugger != nil {
			//If no one is listening to testInfo, dont calculate it
			if len(testInfo) < cap(testInfo) {

				if len(status) < cap(status) {
					status <- fmt.Sprintf("Starting testing of epoch %d", n)
				}

				accuracy, loss, err := t.Test()
				if err != nil {
					return err
				}

				/*
				     fmt.Printf("\rTrainAccuracy:%.4f TrainLoss:%.6f TestAccuracy:%.4f --- \n",
				     float64(epochCorrectTrains)/float64(t.data.TrainSet.GetSetSize()),
				     epochTrainLoss/float64(t.data.TrainSet.GetSetSize()),
				     accuracy,
				   )
				*/

				testInfo <- &debug.TestInfo{
					Epoch:    n,
					Loss:     loss,
					Accuracy: accuracy,
				}

			}
		}

		//Reset to start new epoch
		t.data.TrainSet.Reset()
	}

	if len(status) < cap(status) {
		status <- "Finished"
	}

	return nil
}

func (t *BPTrainer) Test() (accuracy, loss float64, err error) {

	ds := t.data.TestSet

	ds.Reset()
	n := ds.GetSetSize()
	ncorrect := 0
	totalloss := 0.0
	for i := 0; i < n; i++ {
		input, lbl, e := ds.GetNextSet()
		if e != nil {
			return 0, 0, err
		}

		out, e := t.net.Activate(input)
		if err != nil {
			return 0, 0, err
		}

		if ds.IsAnswer(out, lbl) {
			ncorrect++
		}

		totalloss += t.costFunction.Cost(out, lbl)

	}

	accuracy = float64(ncorrect) / float64(n)
	loss = totalloss / float64(n)
	err = nil

	return
}
