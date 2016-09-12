package tensor

import "errors"

//Tensor is a structure to represent a multiple dimension matrix
type Tensor struct {
	Values []float64
	Size   []int //Size of each dimension in order. Values should be bigger than 0

	tmpStrides []int
}

func NewTensor(size ...int) *Tensor {
	t := &Tensor{}
	err := t.Allocate(size...)
	if err != nil {
		panic(err)
	}
	return t
}

//GetDims returns the number of dimensions
func (t *Tensor) GetDims() int {
	return len(t.Size)
}

//GetVal return the value at the index
func (t *Tensor) GetVal(index ...int) float64 {
	return t.Values[t.DimToFlat(index...)]
}

//SetVal sets the value at index to val
func (t *Tensor) SetVal(val float64, index ...int) {
	t.Values[t.DimToFlat(index...)] = val
}

//AddVal increases the value at index by val
func (t *Tensor) AddVal(val float64, index ...int) {
	//fmt.Printf("adding %.4f to %v. size: %v \n", val, index, t.size)
	t.Values[t.DimToFlat(index...)] += val
}

//Zero sets all values to val
func (t *Tensor) Zero(val float64) {
	n := t.GetNumberOfValues()
	for i := 0; i < n; i++ {
		t.Values[i] = val
	}
}

func (t *Tensor) InBounds(index ...int) bool {
	//Check that the dimensions of index match the dimensions of the data
	if len(index) != len(t.Size) {
		panic("Index dimensions do not match Tensor dimensions")
	}
	for i := 0; i < len(index); i++ {
		if index[i] < 0 || index[i] >= t.Size[i] {
			return false
		}
	}

	return true
}

/* DiToFlat returns the position in the Values slice of a value in the position given by the index.

For example in a 2x2 tensor, DimToFlat(0,0) will return 0 and DimToFlat(0,1) will return 2
*/
func (t *Tensor) DimToFlat(index ...int) int {
	//Check that the dimensions of index match the dimensions of the data
	if len(index) != len(t.Size) {
		panic("Index dimensions do not match Tensor dimensions")
	}
	for i := 0; i < len(index); i++ {
		if index[i] < 0 || index[i] >= t.Size[i] {
			panic("Index out of bounds")
		}
	}

	/*The idea behind this algorithm:
	  pos is the position we want to calculate
	  siz is the size of the previous dimensions

	  As the data is stored linearly, we have to map the coordinates in n dimensions to 1 dimension.
	  In the first iteration (x dimension) we simply increase pos by x.
	  In the seond iteration (y dimension) we increase by y * sizeOfXDimension
	  In the third iteration (z dimension) we increase by z * (sizeOfXDimension * sizeOfYDimension)
	  ...etc
	*/

	//Optimitzations
	if len(index) == 1 {
		return index[0]
	}

	if len(index) == 2 {
		return index[0] + index[1]*t.Size[0]
	}

	//Generic
	if len(t.tmpStrides) != len(t.Size) {
		//If we haven't calculated the strides, do it
		t.calcTmpStrides()
	}

	pos := 0
	for i := 0; i < len(index); i++ {
		pos += t.tmpStrides[i] * index[i]
	}

	return pos

}

func (t *Tensor) FlatToDim(index int) []int {
	if index < 0 || index >= len(t.Values) {
		panic("Index out of bounds")
	}

	if len(t.tmpStrides) != len(t.Size) {
		//If we haven't calculated the strides, do it
		t.calcTmpStrides()
	}

	ret := make([]int, len(t.Size))
	for i := len(t.Size) - 1; i >= 0; i-- {
		ret[i] = index / t.tmpStrides[i]
		index = index % t.tmpStrides[i]
	}

	return ret
}

//Copy the values into a new Tensor struct
func (t *Tensor) Copy() *Tensor {
	size := make([]int, len(t.Size))
	data := make([]float64, len(t.Values))

	c := &Tensor{}

	copy(size, t.Size)
	copy(data, t.Values)

	c.Size = size
	c.Values = data

	return c
}

//SetSize sets the size of Tensor and allocates the necessary memory
func (t *Tensor) Allocate(size ...int) error {
	if len(size) == 0 {
		return errors.New("Allocate expects at least one dimension")
	}

	for i := 0; i < len(size); i++ {
		if size[i] <= 0 {
			return errors.New("Allocating a slice with a dimension of size zero or negative is not allowed")
		}
	}

	t.Size = size
	t.Values = make([]float64, SizeLength(size))
	t.calcTmpStrides()

	return nil
}

func (t *Tensor) calcTmpStrides() {
	t.tmpStrides = make([]int, len(t.Size))
	siz := 1
	for i := 0; i < len(t.tmpStrides); i++ {
		t.tmpStrides[i] = siz
		siz *= t.Size[i]
	}
}

func (t *Tensor) Substract(s *Tensor) error {
	if !s.HasSize(t.Size) {
		return errors.New("Cannot substract tensors of different shape/size")
	}

	for i := range t.Values {
		t.Values[i] -= s.Values[i]
	}

	return nil
}

func (t *Tensor) Add(a ...*Tensor) error {
	for i := 0; i < len(a); i++ {
		if !a[i].HasSize(t.Size) {
			return errors.New("Cannot add tensors of different shape/size")
		}
	}

	for i := range t.Values {
		for n := range a {
			t.Values[i] += a[n].Values[i]
		}
	}

	return nil
}

func (t *Tensor) Mul(s float64) {
	for i := range t.Values {
		t.Values[i] *= s
	}
}

func (t *Tensor) GetNumberOfValues() int {
	return SizeLength(t.Size)
}

func (t *Tensor) HasSize(size []int) bool {
	if len(t.Size) != len(size) {
		return false
	}
	for i := range t.Size {
		if t.Size[i] != size[i] {
			return false
		}
	}

	return true
}

//Utils
func SizeLength(size []int) int {
	ret := 1
	for i := range size {
		if size[i] <= 0 {
			panic("Size slice must have values bigger than 0")
		}
		ret *= size[i]
	}

	return ret
}
