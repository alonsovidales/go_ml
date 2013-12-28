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
		[]string{"test_data/nn/theta1.csv", "test_data/nn/theta2.csv"},
		//[]string{},
	)

	x, fx, i := fmincgNn(1, nn, 200)

	fmt.Println("X:", x)
	fmt.Println("fx:", fx)
	fmt.Println("i:", i)

	//nn.InitializeThetas([]int{400, 50, 10})

	/*finalCost, performance := nn.MinimizeCost(30, 1, false, true)
	fmt.Println("Final Cost:", finalCost)
	fmt.Println("Performance:", performance)*/

	/*j, _, _ := nn.NeuralNetCostFunction(0)
	fmt.Println(j)

	if finalCost != 0.3356995121261227 {
		t.Error("The expected cost is 0.3356995121261227, but ", finalCost, "obtained")
	}*/
}
