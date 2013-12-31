/*
Linear Regression implementation
*/
package ml

import (
	"github.com/alonsovidales/matrix"
	//"math"
	//"time"
	//"math/rand"
	"fmt"
	"strconv"
	"io/ioutil"
	"strings"
)

type LinReg struct {
	X [][]float64 // Training set of values for each feature, the first dimension are the test cases
	Y []float64 // The training set with values to be predicted
	// 1st dim -> layer, 2nd dim -> neuron, 3rd dim theta
	Theta []float64
}

func (lr *LinReg) rollThetasGrad(x [][][]float64) [][]float64 {
	return x[0]
}

func (lr *LinReg) unrollThetasGrad(x [][]float64) [][][]float64 {
	return [][][]float64{
		x,
	}
}

func (lr *LinReg) setTheta(t [][][]float64) {
	lr.Theta = t[0][0]
	fmt.Println("setTheta", lr.Theta)
}

func (lr *LinReg) getTheta() [][][]float64 {
	fmt.Println("getTheta", lr.Theta)
	return [][][]float64{
		[][]float64{
			lr.Theta,
		},
	}
}

func (lr *LinReg) CostFunction(lambda float64, calcGrad bool) (j float64, grad [][][]float64, err error) {
	if len(lr.Y) != len(lr.X) {
		err = fmt.Errorf(
			"The number of test cases (X) %d doesn't corresponds with the number of values (Y) %d",
			len(lr.X),
			len(lr.Y))
		return
	}

	if len(lr.Theta) != len(lr.X[0]) {
		err = fmt.Errorf(
			"The Theta arg has a lenght of %d and the input data %d",
			len(lr.Theta),
			len(lr.X[0]))
		return
	}

	m := float64(len(lr.X))
	y := [][]float64{lr.Y}
	theta := [][]float64{lr.Theta}

	pred := mt.Trans(mt.Mult(lr.X, mt.Trans(theta)))
	errors := mt.SumAll(mt.Apply(mt.Sub(pred, y), powTwo)) / (2 * m)
	regTerm := (lambda / (2 * m)) * mt.SumAll(mt.Apply([][]float64{lr.Theta[1:]}, powTwo))

	j = errors + regTerm
	theta[0][0] = 0
	grad = [][][]float64{mt.Sum(mt.MultBy(mt.Mult(mt.Sub(pred, y), lr.X), 1 / m), mt.MultBy(theta, lambda / m))}

	return
}

func (lr *LinReg) InitializeTheta() {
	lr.Theta = make([]float64, len(lr.X[0]))

	/*rand.Seed(int64(time.Now().Nanosecond()))

	lr.Theta = make([]float64, len(lr.X[0]))
	for i := 0; i < len(lr.X[0]); i++ {
		if rand.Float64() > 0.5 {
			lr.Theta[i] = rand.Float64()
		} else {
			lr.Theta[i] = -rand.Float64()
		}
	}*/
}

// Loads information from the local file located at filePath, and after parse
// it, returns the LinReg ready to be used with all the information loaded
// The file format is:
//      X11 X12 ... X1N Y1
//      X21 X22 ... X2N Y2
//      ... ... ... ... ..
//      XN1 XN2 ... XNN YN
//
// Note: Use a single space as separator
func LoadFile(filePath string) (data *LinReg) {
	strInfo, err := ioutil.ReadFile(filePath)
	if err != nil {
		panic(err)
	}
	data = new(LinReg)

	trainingData := strings.Split(string(strInfo), "\n")
	for _, line := range trainingData {
		if line == "" {
			break
		}

		var values []float64
		for _, value := range strings.Split(line, " ") {
			floatVal, err := strconv.ParseFloat(value, 64)
			if err != nil {
				panic(err)
			}
			values = append(values, floatVal)
		}
		data.X = append(data.X, values[:len(values)-1])
		data.Y = append(data.Y, values[len(values)-1])
	}

	return
}
