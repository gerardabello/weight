package layers

import (
	"testing"

	"gitlab.com/gerardabello/weight/marshaling"
	"gitlab.com/gerardabello/weight/tensor"
)

func TestGraphActivation(t *testing.T) {

	g := NewFFNet()

	dl1 := NewDenseLayer([]int{6}, []int{5})
	err := g.AddLayer(dl1)
	if err != nil {
		t.Fatal(err)
	}

	dl2 := NewDenseLayer([]int{5}, []int{2})
	err = g.AddLayer(dl2, dl1.ID())
	if err != nil {
		t.Fatal(err)
	}

	dl3 := NewDenseLayer([]int{5}, []int{2})
	err = g.AddLayer(dl3, dl1.ID())
	if err != nil {
		t.Fatal(err)
	}

	dl4 := NewDenseLayer([]int{2}, []int{1})
	err = g.AddLayer(dl4, dl2.ID(), dl3.ID())
	if err != nil {
		t.Fatal(err)
	}

	dl5 := NewSigmoidLayer(1)
	err = g.AddLayer(dl5, dl4.ID())
	if err != nil {
		t.Fatal(err)
	}

	g.End()

	in := tensor.NewTensor(6)

	in.SetVal(0.5, 1)

	_, err = g.Activate(in)
	if err != nil {
		t.Fatal(err)
	}

}

func TestMarshalFFNet(t *testing.T) {

	g := NewFFNet()

	dl1 := NewDenseLayer([]int{6}, []int{5})
	err := g.AddLayer(dl1)
	if err != nil {
		t.Fatal(err)
	}

	dl2 := NewDenseLayer([]int{5}, []int{2})
	err = g.AddLayer(dl2, dl1.ID())
	if err != nil {
		t.Fatal(err)
	}

	dl3 := NewDenseLayer([]int{5}, []int{2})
	err = g.AddLayer(dl3, dl1.ID())
	if err != nil {
		t.Fatal(err)
	}

	dl4 := NewDenseLayer([]int{2}, []int{1})
	err = g.AddLayer(dl4, dl2.ID(), dl3.ID())
	if err != nil {
		t.Fatal(err)
	}

	g.End()

	in := tensor.NewTensor(6)

	in.SetVal(0.5, 1)

	result1, err := g.Activate(in)
	if err != nil {
		t.Fatal(err)
	}

	marshaling.MarshalToFile(g, "./test.ffnet")
	g2, err := marshaling.UnmarshalFromFile("./test.ffnet")

	if err != nil {
		t.Fatal(err)
	}

	result2, err := g2.Activate(in)
	if err != nil {
		t.Fatal(err)
	}

	err = result1.Substract(result2)
	if err != nil {
		t.Fatal(err)
	}

	if _, v := result1.MaxAbs(); v > 1e-8 {
		t.Fatal("The results where different after marshaling and unmarshaling")
	}

}
