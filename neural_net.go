/*
Neural network implementation
*/
package ml

import (
	"github.com/alonsovidales/matrix"
	"io/ioutil"
	"math"
	"strconv"
	"strings"
)

type NeuralNet struct {
	X [][]float64 // Training set of values for each feature, the first dimension are the test cases
	Y [][]float64 // The training set with values to be predicted
	// 1st dim -> layer, 2nd dim -> neuron, 3rd dim theta
	Theta [][][]float64
}

func sigmoitNnGrad(z float64) float64 {
	sz := sigmoid(z)

	return sz * (1 - sz)
}

func addBias(m [][]float64) (result [][]float64) {
	result = make([][]float64, len(m))

	for i := 0; i < len(m); i++ {
		result[i] = append([]float64{1}, m[i]...)
	}

	return
}

func oneMinus(x float64) float64 {
	return 1 - x
}

func powTwo(x float64) float64 {
	return math.Pow(x, 2)
}

func sigmoidGradient(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

func removeBias(x [][]float64) (result [][]float64) {
	result = make([][]float64, len(x))
	for i := 0; i < len(x); i++ {
		result[i] = append([]float64{0}, x[0][1:]...)
	}

	return
}

func (data *NeuralNet) NeuralNetCostFunction(lambda float64) (j float64, grad [][]float64, err error) {
	// Calculate the hipotesis for all the layers
	hx := data.X
	for i := 0; i < len(data.Theta); i++ {
		hx = mt.Apply(mt.Mult(addBias(hx), mt.Trans(data.Theta[i])), sigmoid)
	}

	j = mt.SumAll(mt.Sub(
		mt.MultElems(mt.Apply(data.Y, neg), mt.Apply(hx, math.Log)),
		mt.MultElems(mt.Apply(data.Y, oneMinus), mt.Apply(mt.Apply(hx, oneMinus), math.Log)))) / float64(len(data.X))

	// Regularization
	thetaReg := 0.0
	// Remove the bias theta for regularizarion
	for _, theta := range data.Theta {
		auxTheta := make([][]float64, len(theta))
		for i, thetaLine := range theta {
			auxTheta[i] = thetaLine[1:]
		}
		thetaReg += mt.SumAll(mt.Apply(auxTheta, powTwo))
	}
	j += (lambda * thetaReg) / float64(2*len(data.Y))

	// Backpropagation
	tmpGrad := make([][][]float64, len(data.Theta))
	// Initialize the tmpGrad to contain matrix with the same size as thetas
	for i, theta := range data.Theta {
		aux := make([][]float64, len(theta))
		for j := 0; j < len(theta); j++ {
			aux[j] = make([]float64, len(theta[0]))
		}
		tmpGrad[i] = aux
	}
	for i := 0; i < len(data.X); i++ {
		// FW
		a := make([][][]float64, len(data.Theta)+1)
		a[0] = addBias([][]float64{data.X[i]})
		z := make([][][]float64, len(data.Theta))
		for i := 0; i < len(data.Theta); i++ {
			z[i] = mt.Mult(a[i], mt.Trans(data.Theta[i]))
			a[i+1] = addBias(mt.Apply(z[i], sigmoid))
		}

		// BW
		delta := make([][][]float64, len(data.Theta))

		delta[len(data.Theta)-1] = mt.Sub([][]float64{a[len(data.Theta)][0][1:]}, [][]float64{data.Y[i]})
		for d := len(data.Theta) - 1; d > 0; d-- {
			delta[d-1] = mt.MultElems(mt.Mult(delta[d], data.Theta[d]), addBias(mt.Apply(z[d-1], sigmoidGradient)))

			tmpGrad[d-1] = mt.Sum(tmpGrad[d-1], mt.Mult(mt.Trans([][]float64{delta[d-1][0][1:]}), a[d-1]))
		}
	}

	grad = make([][]float64, len(data.Theta))
	for i := 0; i < len(tmpGrad); i++ {
		grad[i] = mt.Sum(mt.MultBy(tmpGrad[i], 1/float64(len(data.X))), mt.MultBy(removeBias(data.Theta[i]), lambda/float64(len(data.X))))[0]
	}

	return
}

func NewNeuralNetFromCsv(xSrc string, ySrc string, thetaSrc []string) (result *NeuralNet) {
	result = new(NeuralNet)

	// Parse the X params
	strInfo, err := ioutil.ReadFile(xSrc)
	if err != nil {
		panic(err)
	}

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
		result.X = append(result.X, values)
	}

	// Parse the Y params
	strInfo, err = ioutil.ReadFile(ySrc)
	if err != nil {
		panic(err)
	}

	trainingData = strings.Split(string(strInfo), "\n")
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
		result.Y = append(result.Y, values)
	}

	// Parse the Theta params
	for _, thetaNSrc := range thetaSrc {
		strInfo, err = ioutil.ReadFile(thetaNSrc)
		if err != nil {
			panic(err)
		}

		trainingData = strings.Split(string(strInfo), "\n")
		theta := [][]float64{}
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
			theta = append(theta, values)
		}

		result.Theta = append(result.Theta, theta)
	}

	return
}
