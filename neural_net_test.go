package ml

import (
	"testing"
	"fmt"
)

func TestRollUnroll(t *testing.T) {
	nn := NewNeuralNetFromCsv(
		"test_data/nn/x.csv",
		"test_data/nn/y.csv",
		[]string{},
	)

	nn.InitializeThetas([]int{400, 50, 10})

	initTheta := make([][][]float64, len(nn.Theta))

	for i := 0; i < len(nn.Theta); i++ {
		initTheta[i] = make([][]float64, len(nn.Theta[i]))
		for j := 0; j < len(nn.Theta); j++ {
			initTheta[i][j] = make([]float64, len(nn.Theta[i][j]))
			for k := 0; k < len(nn.Theta); k++ {
				initTheta[i][j][k] = nn.Theta[i][j][k]
			}
		}
	}

	finalTheta := nn.unrollThetasGrad(nn.rollThetasGrad(nn.Theta))

	for i := 0; i < len(nn.Theta); i++ {
		for j := 0; j < len(nn.Theta); j++ {
			for k := 0; k < len(nn.Theta); k++ {
				if finalTheta[i][j][k] != initTheta[i][j][k] {
					t.Error("Theta val after roll and unroll: ", finalTheta[i][j][k], "init theta:", initTheta[i][j][k], "pos:", i, j, k)
				}
			}
		}
	}

}

// Using a predefined dataset stored in test_data/test_linear.dat , calculate
// the cost and gradient for different lambda y theta
func TestNeuralNet(t *testing.T) {
	nn := NewNeuralNetFromCsv(
		"test_data/nn/x.csv",
		"test_data/nn/y.csv",
		[]string{"test_data/nn/initial_Theta1.csv", "test_data/nn/initial_Theta2.csv"},
	)

	//nn.InitializeThetas([]int{400, 25, 10})

	fx, i := nn.fmincgNn(1, 3, false)

	if i != 0 {
		t.Error("Some error happened on fmincgNn calculation :'(")
	}

	if fx[0] > 3.3 {
		t.Error("Expected J too hight on first iteration:", fx[0])
	}

	if fx[1] > 3.25 {
		t.Error("Expected J too hight on first iteration:", fx[1])
	}

	if fx[2] > 3.21 {
		t.Error("Expected J too hight on first iteration:", fx[2])
	}
}

func TestNeuralNetMinimizeCost(t *testing.T) {
	nn := NewNeuralNetFromCsv(
		"test_data/nn/x.csv",
		"test_data/nn/y.csv",
		[]string{},
	)

	nn.InitializeThetas([]int{400, 25, 10})
	j, performance := nn.MinimizeCost(30, true, true)

	fmt.Println("J:", j)
	fmt.Println("Perf:", performance)
	if j > 0.8 {
		t.Error("Cost bigger than 0.8:", j)
	}
	if performance < 0.8 {
		t.Error("Performance worstest than 0.8:", performance)
	}
}
