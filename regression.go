package ml

import (
    "fmt"
    "math"
    "io/ioutil"
    "strings"
    "strconv"
)

type DataSet struct {
    X [][]float64 // Training set of values for each feature
    Y []float64 // The training set with values to be predicted
}

// Calculates the cost function and gradient for the data with the given theta
// and lambda 
func (data *DataSet) LinearRegCostFunction(theta []float64, lambda float64) (j float64, grad []float64, err error) {
    if len(data.Y) != len(data.X) {
        err = fmt.Errorf(
            "The number of test cases (X) %d doesn't corresponds with the number of values (Y) %d",
            len(data.X),
            len(data.Y))
        return
    }

    if len(theta) != len(data.X[0]) {
        err = fmt.Errorf(
            "The Theta arg has a lenght of %d and the input data %d",
            len(theta),
            len(data.X[0]))
        return
    }

    m := len(data.X)
    mF := float64(m)
    // Calculate the hipotesis for each training val
    hipotesis := make([]float64, m)

    for i := 0; i < m; i++ {
        for j:= 0; j < len(theta); j++ {
            hipotesis[i] += data.X[i][j] * theta[j]
        }
    }

    cost := 0.0

    grad = make([]float64, len(theta))
    // Calculate the sum of the square of distances between the wanted result
    // and the actual one, the total cost
    for i := 0; i < len(hipotesis); i++ {
        cost += math.Pow(hipotesis[i] - data.Y[i], 2)

        for j:= 0; j < len(theta); j++ {
            grad[j] += (hipotesis[i] - data.Y[i]) * data.X[i][j]
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

// Loads information from the local file located at filePath, and after parse
// it, returns the DataSet ready to be used with all the information loaded
// The file format is:
//      X11 X12 ... X1N Y1
//      X21 X22 ... X2N Y2
//      ... ... ... ... ..
//      XN1 XN2 ... XNN YN
//
// Note: Use a single space as separator
func LoadFile(filePath string) (data *DataSet) {
    strInfo, err := ioutil.ReadFile(filePath)
    if err != nil { panic(err) }
    data = new(DataSet)

    trainingData := strings.Split(string(strInfo), "\n")
    for _, line := range trainingData {
        if line == "" {
            break
        }

        var values []float64
        for _, value := range strings.Split(line, " ") {
            floatVal, err := strconv.ParseFloat(value, 64)
            if err != nil { panic(err) }
            values = append(values, floatVal)
        }
        data.X = append(data.X, values[:len(values) - 1])
        data.Y = append(data.Y, values[len(values) - 1])
    }

    return
}
