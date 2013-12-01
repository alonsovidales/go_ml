package ml

import (
    "testing"
    "fmt"
)

func TestLinearRegCostFunction(t *testing.T) {

    // Data obtained from the "Andrew Ng" machine learning course from Coursera
    // https://www.coursera.org/course/ml
    x := [][]float64 {
	[]float64{1, -15.9368},
	[]float64{1, -29.1530},
	[]float64{1, 36.1895},
	[]float64{1, 37.4922},
	[]float64{1, -48.0588},
	[]float64{1, -8.9415},
	[]float64{1, 15.3078},
	[]float64{1, -34.7063},
	[]float64{1, 1.3892},
	[]float64{1, -44.3838},
	[]float64{1, 7.0135},
	[]float64{1, 22.7627},
    }

    y := []float64{
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
    }

    expectedJ := 303.9937949908333
    expectedGrad := [2]float64{-15.303049999999999, 598.251293035}

    data := &DataSet{
	x: x[:],
	y: y,
    }

    j, grad, err := data.LinearRegCostFunction([]float64{1, 1}, 1)

    // The expected results are:
    //	J: 303.99
    //	Grad: [-15.303, 598.251]

    if err != nil {
	t.Error("The LinearRegCostFunction returned an error:", err)
    }

    if (j != expectedJ) {
	t.Error("The expected value for J is:", expectedJ, "but the returned value is:", j)
    }

    for i := 0; i < len(grad); i++ {
	if (grad[i] != expectedGrad[i]) {
	    t.Error("The expected gradient is:", expectedGrad[i], "but the returned value is:", grad[i])
	}
    }

    fmt.Println("J:", j)
    fmt.Println("Grad:", grad)
    fmt.Println("Err:", err)
}
