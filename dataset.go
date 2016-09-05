package weight

import "gitlab.com/gerardabello/weight/tensor"

//PairSet binds a Train and a Test Set together.
type PairSet struct {
	TrainSet DataSet
	TestSet  DataSet
}

//Close closes both sets
func (ps *PairSet) Close() {
	ps.TrainSet.Close()
	ps.TestSet.Close()
}

//DataSet is an interface that returns neural net inputs and tells you if the outputs are correct.
type DataSet interface {
	GetDataSize() []int
	GetAnswersSize() []int
	GetSetSize() int

	//GetNextSet returns an input, the desired answer, and and error
	GetNextSet() (*tensor.Tensor, *tensor.Tensor, error)

	//After a reset, NextSet will be the first
	Reset()

	Close()

	//IsAnswer returns true if the output is to be considered correct based on the correct answer. For example, in classification data sets, this should return true if the maximum probability in output and answer is in the same label.
	IsAnswer(output *tensor.Tensor, answer *tensor.Tensor) bool
}

//TestLayer return accuracy for a given layer and a given DataSet
func TestLayer(layer Layer, ds DataSet) (float64, error) {
	ds.Reset()
	n := ds.GetSetSize()
	ncorrect := 0
	for i := 0; i < n; i++ {
		input, lbl, err := ds.GetNextSet()
		if err != nil {
			return 0, err
		}

		out, err := layer.Activate(input)
		if err != nil {
			return 0, err
		}

		if ds.IsAnswer(out, lbl) {
			ncorrect++
		}

	}

	return float64(ncorrect) / float64(n), nil
}
