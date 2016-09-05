package layers

import (
	"testing"

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

	g.End()

	in := tensor.NewTensor(6)

	in.SetVal(0.5, 1)

	_, err = g.Activate(in)
	if err != nil {
		t.Fatal(err)
	}

}
