package ml

import (
    "testing"
    "fmt"
)

// Using a predefined dataset stored in test_data/test_logistic.dat , calculate
// the cost and gradient for different lambda y theta
func TestLinearRegCostFunction(t *testing.T) {

    // Data obtained from the "Andrew Ng" machine learning course from Coursera
    // https://www.coursera.org/course/ml
    data := LoadFile("test_data/test_logistic.dat")

    theta := [][]float64 {
        []float64{1, 1},
        []float64{13.08790, 0.36778},
    }
    lambda := []float64 {
        1,
        0,
    }

    expectedJ := []float64 {
        303.9937949908333,
        22.3738964254415,
    }

    expectedGrad := [][]float64 {
        []float64{-15.303049999999999, 598.251293035},
        []float64{-3.2069500000601416e-05, 0.0009341162152152194},
    }

    for test := 0; test < len(expectedGrad); test++ {
        j, grad, err := data.LinearRegCostFunction(theta[test], lambda[test])

        // The expected results are:
        //        J: 303.99
        //        Grad: [-15.303, 598.251]

        if err != nil {
            t.Error("The LinearRegCostFunction returned an error:", err)
        }

        if (j != expectedJ[test]) {
            t.Error("The expected value for J is:", expectedJ[test], "but the returned value is:", j)
        }

        for i := 0; i < len(grad); i++ {
            if (grad[i] != expectedGrad[test][i]) {
                t.Error("The expected gradient is:", expectedGrad[test][i], "but the returned value is:", grad[i])
            }
        }
    }
}

func TestCalculateOptimumData(t *testing.T) {

    // Data obtained from the "Andrew Ng" machine learning course from Coursera
    // https://www.coursera.org/course/ml
    //data := LoadFile("test_data/test_logistic_polynomial.dat")
    data := LoadFile("test_data/test_logistic.dat")

    fmt.Println("Lambda", data.CalcOptimumLambdaTheta())
}
