package ml

import (
	"testing"
	"fmt"
)

// Using a predefined dataset stored in test_data/test_linear.dat , calculate
// the cost and gradient for different lambda y theta
func TestLinearRegCostFunction(t *testing.T) {

	// Data obtained from the "Andrew Ng" machine learning course from Coursera
	// https://www.coursera.org/course/ml
	data := LoadFile("test_data/test_linear.dat")
	data.LinearReg = true

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
		[]float64{-15.219716666666665, 598.251293035},
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

func TestCalculateOptimumTheta(t *testing.T) {

	// Data obtained from the "Andrew Ng" machine learning course from Coursera
	// https://www.coursera.org/course/ml
	data := &Regression {
		LinearReg: true,
		X: [][]float64{
			[]float64{1.0000, -15.9368},
			[]float64{1.0000, -29.1530},
			[]float64{1.0000, 36.1895},
			[]float64{1.0000, 37.4922},
			[]float64{1.0000, -48.0588},
			[]float64{1.0000, -8.9415},
			[]float64{1.0000, 15.3078},
			[]float64{1.0000, -34.7063},
			[]float64{1.0000, 1.3892},
			[]float64{1.0000, -44.3838},
			[]float64{1.0000, 7.0135},
			[]float64{1.0000, 22.7627},
		},
		Y: []float64{
			2.1343,
		        1.1733,
			34.3591,
			36.8380,
			2.8090,
			2.1211,
			14.7103,
			2.6142,
			3.7402,
			3.7317,
			7.6277,
			22.7524,
		},
	}
	data.InitializeTheta()

	Fmincg(data, 0.0, 10, true)
	j, _, _ := data.CostFunction(0.0, false)

	if j != 22.373896424566116 {
		t.Error("The expected cost value is 22.373896424566116, but:", j, "obtained")
	}
	if data.Theta[0] != 13.087927305447673 || data.Theta[1] != 0.3677790632076952 {
		t.Error("The expected theta values are 13.087927305447673 and 0.3677790632076952, but:", data.Theta, "obtained")
	}
}

func TestLogisticHipotesis(t *testing.T) {
	data := &Regression{
		Theta: []float64{-25.161272, 0.206233, 0.201470},
		LinearReg: false,
	}
	h := data.LogisticHipotesis([]float64{1, 45, 85})
	fmt.Println("Hip:", h)
	if h != 0.7762878133064746 {
		t.Error("The expected value is 0.7762878133064746, but the returned value is:", h)
	}
}

func TestCalculateOptimumDataLogRegWithPrepare(t *testing.T) {
	// Data obtained from the "Andrew Ng" machine learning course from Coursera
	// https://www.coursera.org/course/ml
	data := LoadFile("test_data/test_log_regression_2.dat")
	data.LinearReg = false
	data.X = MapFeatures(data.X, 9)
	data.InitializeTheta()

	j, _, _ := data.MinimizeCost(200, true, true)

	if j > 0.4 {
		t.Error("The expected performance is better than: 0.4 but the returned value is:", j)
	}
}
