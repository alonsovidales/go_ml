package ml

import (
	"testing"
)

// Using a predefined dataset stored in test_data/test_linear.dat , calculate
// the cost and gradient for different lambda y theta
func TestLinearRegCostFunction(t *testing.T) {

	// Data obtained from the "Andrew Ng" machine learning course from Coursera
	// https://www.coursera.org/course/ml
	data := LoadFile("test_data/test_linear.dat")

	theta := [][]float64{
		[]float64{1, 1},
		[]float64{13.08790, 0.36778},
	}
	lambda := []float64{
		1,
		0,
	}

	expectedJ := []float64{
		303.9937949908333,
		22.3738964254415,
	}

	expectedGrad := [][]float64{
		[]float64{-15.303049999999999, 598.251293035},
		[]float64{-3.2069500000601416e-05, 0.0009341162152152194},
	}

	for test := 0; test < len(expectedGrad); test++ {
		data.Theta = theta[test]
		j, grad, err := data.CostFunction(lambda[test], true)

		if err != nil {
			t.Error("The LinearRegCostFunction returned an error:", err)
		}

		if j != expectedJ[test] {
			t.Error("The expected value for J is:", expectedJ[test], "but the returned value is:", j)
		}

		for i := 0; i < len(grad[0][0]); i++ {
			if grad[0][0][i] != expectedGrad[test][i] {
				t.Error("The expected gradient is:", expectedGrad[test][i], "but the returned value is:", grad[0][0][i])
			}
		}
	}
}

/*
func TestCalculateOptimumTheta(t *testing.T) {

	// Data obtained from the "Andrew Ng" machine learning course from Coursera
	// https://www.coursera.org/course/ml
	data := LoadFile("test_data/test_linear_polynomial.dat")
	data.InitializeTheta()

	Fmincg(data, 0.01, 10, true)
	j, _, _ := data.CostFunction(0.01, false)
	fmt.Println("Theta:", data.Theta)
	fmt.Println("J:", j)

	if performance != 4.901045349988667 {
		t.Error("The expected performance is: 4.901045349988667 but the returned value is:", performance)
	}

	if lambda != 0.1 {
		t.Error("The expected lambda is: 0.1 but the returned value is:", lambda)
	}

	expectedTheta := []float64{
		11.668527097769664,
		9.227365853102674,
		5.901606532678573,
		2.732609803669128,
		2.048572624793663,
		1.417634232412458,
		-0.23465057648197382,
		0.5051641750629966,
		0.43653605626382835,
	}

	for i := 0; i < len(expectedTheta); i++ {
		if theta[i] != expectedTheta[i] {
			t.Error("The expected theta value on", i, " is:", expectedTheta[i], "but the returned value was:", theta[i])
		}
	}
}*/
