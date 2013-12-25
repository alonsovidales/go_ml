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
	)

	fmt.Println("X:", len(nn.X))
	fmt.Println("Y:", len(nn.Y))
	fmt.Println("Theta:", len(nn.Theta))

	cost, _, _ := nn.NeuralNetCostFunction(0.5)
	fmt.Println("Cost:", cost)

	if cost != 0.3356995121261227 {
		t.Error("The expected cost is 0.3356995121261227, but ", cost, "obtained")
	}
}
