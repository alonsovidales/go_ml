package ml

import (
    "fmt"
    "math"
    "io/ioutil"
    "strings"
    "strconv"
    "math/rand"
    "time"
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
        thetaReg += math.Pow(theta[i], 2)
    }

    // Calculate the regularized cost
    j = (cost / float64(2 * m)) + ((lambda * thetaReg) / float64(2 * m))

    grad[0] /= mF
    for j:= 1; j < len(theta); j++ {
        grad[j] = (grad[j] / mF) + ((lambda * theta[j]) / mF)
    }

    return
}

func (data *DataSet) MinimizeTheta(initTheta []float64, lambda float64, maxIters int) (theta []float64) {
    theta = initTheta

    // We will try to adjust this alpha to the best possible on the first iterations
    alpha := 1.0

    jTraining, _, err := data.LinearRegCostFunction(theta, lambda)
    if err != nil { panic(err) }
    lastJ := jTraining + 1
    lastTheta := make([]float64, len(initTheta))

    for iter := 0; iter < maxIters; iter++ {
        jTraining, grad, err := data.LinearRegCostFunction(theta, lambda)
        if err != nil { panic(err) }

        if jTraining > lastJ {
            alpha /= 10

            lastJ = jTraining
            copy(theta, lastTheta)
        } else {
            copy(lastTheta, theta)
            for j := 0; j < len(theta); j++ {
                theta[j] -= alpha * grad[j]
            }

            if lastJ - jTraining < 0.001 {
                return
            }

            lastJ = jTraining
        }
    }

    return
}

func (data *DataSet) shuffle() (shuffledData *DataSet) {
    aux := make([][]float64, len(data.X))

    copy(aux, data.X)
    for i := 0; i < len(aux); i++ {
        aux[i] = append(aux[i], data.Y[i])
    }

    dest := make([][]float64, len(aux))
    rand.Seed(int64(time.Now().Nanosecond()))
    perm := rand.Perm(len(aux))
    for i, v := range perm {
        dest[v] = aux[i]
    }

    shuffledData = &DataSet{
        X: dest,
        Y: make([]float64, len(dest)),
    }
    for i := 0; i < len(aux); i++ {
        shuffledData.Y[i] = shuffledData.X[i][len(shuffledData.X[i]) - 1]
        shuffledData.X[i] = shuffledData.X[i][:len(shuffledData.X[i]) - 1]
    }

    return
}

func (data *DataSet) CalcOptimumLambdaTheta(maxIters int, suffleData bool) (lambda float64, theta []float64, performance float64) {
    lambdas := []float64{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}

    // Get the 60% of the data as training data, 20% as cross validation, and
    // the remaining 20% as test data
    training := int64(float64(len(data.X)) * 0.6)
    cv := int64(float64(len(data.X)) * 0.8)

    if suffleData {
        data = data.shuffle()
    }

    trainingData := &DataSet{
        X: data.X[:training],
        Y: data.Y[:training],
    }
    cvData := &DataSet{
        X: data.X[training:cv],
        Y: data.Y[training:cv],
    }
    testData := &DataSet{
        X: data.X[cv:],
        Y: data.Y[cv:],
    }

    bestLambda := 0.0
    posTheta := trainingData.MinimizeTheta(make([]float64, len(trainingData.X[0])), bestLambda, maxIters)
    bestTheta := posTheta
    jCv, _, err := cvData.LinearRegCostFunction(posTheta, bestLambda)
    if err != nil { panic(err) }
    lowerJ := jCv
    for _, posLambda := range lambdas {
        posTheta := trainingData.MinimizeTheta(make([]float64, len(trainingData.X[0])), posLambda, maxIters)
        jCv, _, err := cvData.LinearRegCostFunction(posTheta, 0)
        if err != nil { panic(err) }

        if jCv < lowerJ {
            bestTheta = posTheta
            lowerJ = jCv
            bestLambda = posLambda
        }
    }

    lambda = bestLambda
    theta = bestTheta
    performance, _, _ = testData.LinearRegCostFunction(theta, 0)

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
