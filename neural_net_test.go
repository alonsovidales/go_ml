package ml

import (
	"testing"
	"fmt"
)

// Using a predefined dataset stored in test_data/test_linear.dat , calculate
// the cost and gradient for different lambda y theta
func TestNeuralNet(t *testing.T) {
	//t.Error("The expected gradient is:", expectedGrad[test][i], "but the returned value is:", grad[i])

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
