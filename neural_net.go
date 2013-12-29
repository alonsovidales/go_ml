/*
Neural network implementation
*/
package ml

import (
	"github.com/alonsovidales/matrix"
	"io/ioutil"
	"math"
	"strconv"
	"math/rand"
	"time"
	"fmt"
	"strings"
)

type NeuralNet struct {
	X [][]float64 // Training set of values for each feature, the first dimension are the test cases
	Y [][]float64 // The training set with values to be predicted
	// 1st dim -> layer, 2nd dim -> neuron, 3rd dim theta
	Theta [][][]float64
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
		result[i] = append([]float64{0}, x[i][1:]...)
	}

	return
}

func (nn *NeuralNet) rollThetasGrad(x [][][]float64) [][]float64 {
	result := []float64{}
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x[i][0]); j++ {
			for k := 0; k < len(x[i]); k++ {
				result = append(result, x[i][k][j])
			}
		}
	}

	return [][]float64{result}
}

func (nn *NeuralNet) unrollThetasGrad(x [][]float64) (r [][][]float64) {
	pos := 0
	r = make([][][]float64, len(nn.Theta))
	for i := 0; i < len(nn.Theta); i++ {
		r[i] = make([][]float64, len(nn.Theta[i]))
		for j := 0; j < len(nn.Theta[i]); j++ {
			r[i][j] = make([]float64, len(nn.Theta[i][j]))
		}
		for j := 0; j < len(nn.Theta[i][0]); j++ {
			for k := 0; k < len(nn.Theta[i]); k++ {
				r[i][k][j] = x[0][pos]
				pos++
			}
		}
	}

	return
}

func (nn *NeuralNet) SaveThetas(targetDir string) (files []string) {
	fileCont := make([]string, len(nn.Theta))
	for i := 0; i < len(nn.Theta); i++ {
		for j := 0; j < len(nn.Theta[i]); j++ {
			s := []string{}
			for k := 0; k < len(nn.Theta[i][j]); k++ {
				s = append(s, strconv.FormatFloat(nn.Theta[i][j][k], 'e', -1, 64))
			}

			fileCont[i] += strings.Join(s, " ") + "\n"
		}
	}

	files = make([]string, len(nn.Theta))
	for i := 0; i < len(nn.Theta); i++ {
		files[i] = fmt.Sprintf("%s/theta_%d.txt", targetDir, i)
		ioutil.WriteFile(
			files[i],
			[]byte(fileCont[i]),
			0644)
	}

	return
}

func (nn *NeuralNet) NeuralNetCostFunction(lambda float64, calcGrad bool) (j float64, grad [][][]float64, err error) {
	// Calculate the hipotesis for all the layers
	hx := nn.X
	for i := 0; i < len(nn.Theta); i++ {
		hx = mt.Apply(mt.Mult(addBias(hx), mt.Trans(nn.Theta[i])), sigmoid)
	}

	j = mt.SumAll(mt.Sub(
		mt.MultElems(mt.Apply(nn.Y, neg), mt.Apply(hx, math.Log)),
		mt.MultElems(mt.Apply(nn.Y, oneMinus), mt.Apply(mt.Apply(hx, oneMinus), math.Log)))) / float64(len(nn.X))

	// Regularization
	thetaReg := 0.0
	// Remove the bias theta for regularizarion
	for _, theta := range nn.Theta {
		auxTheta := make([][]float64, len(theta))
		for i, thetaLine := range theta {
			auxTheta[i] = thetaLine[1:]
		}
		thetaReg += mt.SumAll(mt.Apply(auxTheta, powTwo))
	}
	j += (lambda * thetaReg) / float64(2*len(nn.Y))

	if !calcGrad {
		return
	}

	// Backpropagation
	tmpGrad := make([][][]float64, len(nn.Theta))
	// Initialize the tmpGrad to contain matrix with the same size as thetas
	for i, theta := range nn.Theta {
		tmpGrad[i] = make([][]float64, len(theta))
		for j := 0; j < len(theta); j++ {
			tmpGrad[i][j] = make([]float64, len(theta[0]))
		}
	}
	for i := 0; i < len(nn.X); i++ {
		// FW
		a := make([][][]float64, len(nn.Theta)+1)
		a[0] = addBias([][]float64{nn.X[i]})
		z := make([][][]float64, len(nn.Theta))
		for i := 0; i < len(nn.Theta); i++ {
			z[i] = mt.Mult(a[i], mt.Trans(nn.Theta[i]))
			a[i+1] = addBias(mt.Apply(z[i], sigmoid))
		}

		// BW
		delta := make([][][]float64, len(nn.Theta))
		delta[len(nn.Theta)-1] = mt.Sub([][]float64{a[len(nn.Theta)][0][1:]}, [][]float64{nn.Y[i]})

		for d := len(nn.Theta) - 2; d >= 0; d-- {
			delta[d] = mt.MultElems(mt.Mult(delta[d + 1], nn.Theta[d + 1]), addBias(mt.Apply(z[d], sigmoidGradient)))
			delta[d] = [][]float64{delta[d][0][1:]}
		}

		for d := 0; d < len(tmpGrad); d++ {
			tmpGrad[d] = mt.Sum(tmpGrad[d], mt.Mult(mt.Trans([][]float64{delta[d][0]}), a[d]))
		}
	}

	grad = make([][][]float64, len(nn.Theta))

	tmp := 0.0
	for i := 0; i < len(nn.Theta[0]); i++ {
		tmp += nn.Theta[0][i][0]
	}

	for i := 0; i < len(tmpGrad); i++ {
		grad[i] = mt.Sum(mt.MultBy(tmpGrad[i], 1/float64(len(nn.X))), mt.MultBy(removeBias(nn.Theta[i]), lambda/float64(len(nn.X))))
	}

	return
}

// Random inizialization of the thetas, the layerSizes array will contain on
// each row, the size of the layer to be initialized, the first layer is the
// input nn size, and last layer will correspond to the output layer
func (nn *NeuralNet) InitializeThetas(layerSizes []int) {
	rand.Seed(int64(time.Now().Nanosecond()))
	epsilon := math.Sqrt(6) / math.Sqrt(float64(layerSizes[0] + layerSizes[len(layerSizes) - 1]))

	nn.Theta = make([][][]float64, len(layerSizes) - 1)

	for l := 1; l < len(layerSizes); l++ {
		nn.Theta[l - 1] = make([][]float64, layerSizes[l])
		for n := 0; n < layerSizes[l]; n++ {
			nn.Theta[l - 1][n] = make([]float64, layerSizes[l - 1] + 1)
			for i := 0; i < layerSizes[l - 1] + 1; i++ {
				if rand.Float64() > 0.5 {
					nn.Theta[l - 1][n][i] = (rand.Float64() * epsilon)
				} else {
					nn.Theta[l - 1][n][i] = 0 - (rand.Float64() * epsilon)
				}
			}
		}
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

func (nn *NeuralNet) shuffle() (shuffledData *NeuralNet) {
	aux := make([][]float64, len(nn.X))

	copy(aux, nn.X)

	for i := 0; i < len(aux); i++ {
		aux[i] = append(aux[i], nn.Y[i]...)
	}

	dest := make([][]float64, len(aux))
	rand.Seed(int64(time.Now().Nanosecond()))

	for i, v := range rand.Perm(len(aux)) {
		dest[v] = aux[i]
	}

	shuffledData = &NeuralNet{
		X: make([][]float64, len(nn.X)),
		Y: make([][]float64, len(nn.Y)),
	}
	for i := 0; i < len(dest); i++ {
		shuffledData.Y[i] = dest[i][len(dest[i]) - len(nn.Y[0]):]
		shuffledData.X[i] = dest[i][:len(dest[i]) - len(nn.Y[0])]
	}

	shuffledData.Theta = nn.Theta

	return
}

func (nn *NeuralNet) applyGrad(grad [][][]float64, steep float64) (newTheta [][][]float64) {
	newTheta = make([][][]float64, len(nn.Theta))
	for i := 0; i < len(grad); i++ {
		newTheta[i] = make([][]float64, len(grad[i]))
		for j := 0; j < len(grad[i]); j++ {
			newTheta[i][j] = make([]float64, len(grad[i][j]))
			for k := 0; k < len(grad[i][j]); k++ {
				newTheta[i][j][k] = nn.Theta[i][j][k] - (steep * grad[i][j][k])
			}
		}
	}

	return
}

func copyTheta(theta [][][]float64) (copyTheta [][][]float64) {
	copyTheta = make([][][]float64, len(theta))
	for i := 0; i < len(theta); i++ {
		copyTheta[i] = make([][]float64, len(theta[i]))
		for j := 0; j < len(theta[i]); j++ {
			copyTheta[i][j] = make([]float64, len(theta[i][j]))
			for k := 0; k < len(theta[i][j]); k++ {
				copyTheta[i][j][k] = theta[i][j][k]
			}
		}
	}

	return
}

func (nn *NeuralNet) Hipotesis(x []float64) (result []float64) {
	// Add the bias
	aux := [][]float64{x}

	for _, theta := range nn.Theta {
		aux = mt.Apply(mt.Mult(addBias(aux), mt.Trans(theta)), sigmoid)
	}

	return aux[0]
}

// This metod splits the given nn in three sets: training, cross validation,
// test. In order to calculate the optimal theta, tries with different
// possibilities and the training nn, and check the best match with the cross
// validations, after obtain the best lambda, check the perfomand against the
// test set of nn
func (nn *NeuralNet) MinimizeCost(maxIters int, suffleData bool, verbose bool) (finalCost float64, performance float64) {
	lambdas := []float64{0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300}

	//initTheta := copyTheta(nn.Theta)

	if suffleData {
		nn = nn.shuffle()
	}

	// Get the 60% of the nn as training nn, 20% as cross validation, and
	// the remaining 20% as test nn
	training := int64(float64(len(nn.X)) * 0.6)
	cv := int64(float64(len(nn.X)) * 0.8)

	trainingData := &NeuralNet{
		X: nn.X[:training],
		Y: nn.Y[:training],
		Theta: nn.Theta,
	}

	cvData := &NeuralNet{
		X: nn.X[training:cv],
		Y: nn.Y[training:cv],
		Theta: nn.Theta,
	}
	testData := &NeuralNet{
		X: nn.X[cv:],
		Y: nn.Y[cv:],
		Theta: nn.Theta,
	}

	// Launch a process for each lambda in order to obtain the one with best
	// performance
	bestJ := math.Inf(1)
	bestLambda := 0.0

	for _, posLambda := range lambdas {
		if verbose {
			fmt.Println("Checking Lambda:", posLambda)
		}
		trainingData.fmincgNn(posLambda, 3, verbose)
		cvData.Theta = trainingData.Theta

		j, _, _ := cvData.NeuralNetCostFunction(posLambda, false)

		if bestJ > j {
			bestJ = j
			bestLambda = posLambda
		}
	}

	// Include the cross validation cases into the training for the final train
	trainingData.X = append(trainingData.X, cvData.X...)
	trainingData.Y = append(trainingData.Y, cvData.Y...)

	if verbose {
		fmt.Println("Lambda:", bestLambda)
		fmt.Println("Training with the 80% of the samples...")
	}
	trainingData.fmincgNn(bestLambda, maxIters, verbose)

	testData.Theta = trainingData.Theta
	nn.Theta = trainingData.Theta

	matches := 0.0
	for i := 0; i < len(testData.X); i++ {
		match := true
		prediction := testData.Hipotesis(testData.X[i])

		for i := 0; i < len(prediction); i++ {
			if prediction[i] > 0.5 {
				prediction[i] = 1
			} else {
				prediction[i] = 0
			}
		}

		checkHip: for h := 0; h < len(prediction); h++ {
			if testData.Y[i][h] != prediction[h] {
				match = false
				break checkHip
			}
		}

		if match {
			matches++
		}
	}

	finalCost, _, _ = testData.NeuralNetCostFunction(0, false)

	performance = matches / float64(len(testData.Y))

	return
}
