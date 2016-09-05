# Tensor
Tensor is a multi-dimensional tensor used in the neural network library Weight.

The underlying data is a slice of float64 and the shape is a slice of ints, one for each dimension.

Example usage:
```go
//We can create a tensor manually
t1 := Tensor{
  Values: []float64{
    0, 0, 0,
    1, 0.5, 1,
    0, 1, 0},
  Size: []int{3, 3},
}

//We can also use NewTensor. It creates a tensor with a given size and allocates the Values slice with the required size
t2 := tensor.NewTensor(2,2,1)
//Now we can use any value within the shape of the tensor. This sets the value 0.5 to the position (1,0,0)
t2.SetVal(0.5, 1,0,0)
```
