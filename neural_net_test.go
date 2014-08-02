package ml

import (
	"testing"
	"github.com/alonsovidales/go_matrix"
	"fmt"
)

func TestRollUnroll(t *testing.T) {
	fmt.Println("Testing Roll & Unroll neural networks...")
	nn := NewNeuralNetFromCsv(
		"test_data/nn/x.csv",
		"test_data/nn/y.csv",
		[]string{},
		0,
	)

	nn.InitializeThetas([]int{4, 5, 10})

	initTheta := make([][][]float64, len(nn.Theta))

	for i := 0; i < len(nn.Theta); i++ {
		initTheta[i] = nn.Theta[i].GetMatrixFromCuda()
	}

	nn.setRolledThetas(nn.rollThetasGradTo(nn.Theta, new(mt.CudaMatrix)))

	for i := 0; i < len(nn.Theta); i++ {
		final := nn.Theta[i].GetMatrixFromCuda()
		for r := 0; r < len(final); r++ {
			for c := 0; c < len(final[0]); c++ {
				if initTheta[i][r][c] != final[r][c] {
					t.Error("The expected value in pos:", i, r, c, "is:", initTheta[i][r][c], "but", final[r][c], "obtained.")
				}
			}
		}
	}

}

// Using a predefined dataset stored in test_data/test_linear.dat, calculates
// the cost and gradient for different lambda y theta
func TestNeuralNet(t *testing.T) {
	fmt.Println("Testing Neural Network Fmincg function...")
	nn := NewNeuralNetFromCsv(
		"test_data/nn/x.csv",
		"test_data/nn/y.csv",
		[]string{
			"test_data/nn/initial_Theta1.csv",
			"test_data/nn/initial_Theta2.csv",
		},
		0,
	)

	fx, i, err := Fmincg(nn, 1, 3, true)

	if err != nil {
		t.Error(err)
	}

	if i != 0 {
		t.Error("Some error happened on fmincgNn calculation :'(")
	}

	if fx[0] > 3.3 {
		t.Error("Expected J too hight on first iteration:", fx[0])
	}

	if fx[1] > 3.25 {
		t.Error("Expected J too hight on first iteration:", fx[1])
	}
}

/*func TestNeuralNetMinimizeCost(t *testing.T) {
	fmt.Println("Testing Neural Network training...")

	nn := NewNeuralNetFromCsv(
		"test_data/nn/x.csv",
		"test_data/nn/y.csv",
		[]string{},
	)

	nn.InitializeThetas([]int{400, 25, 10})
	j, performance, _, _ := nn.MinimizeCost(30, true, true)

	if j > 0.8 {
		t.Error("Cost bigger than 0.8:", j)
	}
	if performance < 0.8 {
		t.Error("Performance worstest than 0.8:", performance)
	}
}

func TestNeuralNetSaveLoad(t *testing.T) {
	fmt.Println("Testing Save / Load of thetas for Neural Networks...")

	nn := NewNeuralNetFromCsv(
		"test_data/nn/x.csv",
		"test_data/nn/y.csv",
		[]string{},
	)

	nn.InitializeThetas([]int{400, 25, 10})

	files := nn.SaveThetas("test_data/nn/")

	nn2 := NewNeuralNetFromCsv(
		"test_data/nn/x.csv",
		"test_data/nn/y.csv",
		files,
	)

	j1, _, _ := nn.CostFunction(1, false)
	j2, _, _ := nn2.CostFunction(1, false)

	if j1 != j2 {
		t.Error("The cost returned after store and read the Thetas from files, doesn't match with the initial theta")
	}
}*/
