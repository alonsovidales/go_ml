package ml

import (
    "fmt"
    "math"
)

type DataSet struct {
    x [][]float64 // Training set of values for each feature
    y []float64 // The training set with values to be predicted
}

func (data *DataSet) LinearRegCostFunction(theta []float64, lambda float64) (j float64, grad []float64, err error) {
    if len(data.y) != len(data.x) {
        err = fmt.Errorf(
	    "The number of test cases (X) %d doesn't corresponds with the number of values (Y) %d",
	    len(data.x),
	    len(data.y))
	return
    }

    if len(theta) != len(data.x[0]) {
        err = fmt.Errorf(
	    "The Theta arg has a lenght of %d and the input data %d",
	    len(theta),
	    len(data.x[0]))
	return
    }

    m := len(data.x)
    mF := float64(m)
    // Calculate the hipotesis for each training val
    hipotesis := make([]float64, m)

    for i := 0; i < m; i++ {
        for j:= 0; j < len(theta); j++ {
	    hipotesis[i] += data.x[i][j] * theta[j]
	}
    }

    cost := 0.0

    grad = make([]float64, len(theta))
    // Calculate the sum of the square of distances between the wanted result
    // and the actual one, the total cost
    for i := 0; i < len(hipotesis); i++ {
	cost += math.Pow(hipotesis[i] - data.y[i], 2)

        for j:= 0; j < len(theta); j++ {
	    grad[j] += (hipotesis[i] - data.y[i]) * data.x[i][j]
	}
    }

    thetaReg := 0.0
    for i := 1; i < len(theta); i++ {
	thetaReg += theta[i]
    }

    // Calculate the regularized cost
    j = (cost / float64(2 * m)) + ((lambda * thetaReg) / float64(2 * m))

    grad[0] /= mF
    for j:= 1; j < len(theta); j++ {
	grad[j] = (grad[j] / mF) + ((lambda * theta[j]) / mF)
    }

    return
}
