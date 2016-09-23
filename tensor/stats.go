package tensor

import "math"

type Stats struct {
	Mean  float64
	StDev float64
	Min   float64
	Max   float64
}

func (t *Tensor) Max() (int, float64) {
	index := 0
	largest := t.Values[0]
	n := t.GetNumberOfValues()
	for i := 1; i < n; i++ {
		if t.Values[i] > largest {
			largest = t.Values[i]
			index = i
		}
	}
	return index, largest
}

func (t *Tensor) Min() (int, float64) {
	index := 0
	smallest := t.Values[0]
	n := t.GetNumberOfValues()
	for i := 1; i < n; i++ {
		if t.Values[i] < smallest {
			smallest = t.Values[i]
			index = i
		}
	}
	return index, smallest
}

func (t *Tensor) MaxAbs() (int, float64) {
	index := 0
	largest := t.Values[0]
	n := t.GetNumberOfValues()
	for i := 1; i < n; i++ {
		if math.Abs(t.Values[i]) > math.Abs(largest) {
			largest = t.Values[i]
			index = i
		}
	}
	return index, largest
}

func (t *Tensor) Mean() float64 {
	mean := 0.0
	n := t.GetNumberOfValues()
	for i := 1; i < n; i++ {
		mean += t.Values[i] / float64(n)
	}
	return mean
}

func (t *Tensor) StdDev() float64 {
	if t.GetNumberOfValues() < 2 {
		return 0
	}

	total := 0.0
	mean := t.Mean()
	for _, number := range t.Values {
		total += math.Pow(number-mean, 2)
	}
	variance := total / float64(t.GetNumberOfValues()-1)
	stdev := math.Sqrt(variance)
	return stdev
}

//Stats returns all the statistics about the values in the tensor. It is just a convinient method if you need them all
func (t *Tensor) Stats() *Stats {
	mean := t.Mean()
	stdev := t.StdDev()
	_, min := t.Min()
	_, max := t.Max()
	return &Stats{Mean: mean, StDev: stdev, Min: min, Max: max}
}
