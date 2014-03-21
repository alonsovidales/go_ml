package ml

import (
	"fmt"
	"github.com/alonsovidales/go_matrix"
	"io/ioutil"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

// Neural network representation, the X and Y properties are to be used with
// training proposals
type NeuralNet struct {
	// Training set of values for each feature, the first dimension are the test cases
	X [][]float64
	// The training set with values to be predicted
	Y [][]float64
	// 1st dim -> layer, 2nd dim -> neuron, 3rd dim theta
	Theta [][][]float64
}

// CostFunction Calcualtes the cost function for the training set stored in the
// X and Y properties of the instance, and with the theta configuration.
// The lambda parameter controls the degree of regularization (0 means
// no-regularization, infinity means ignoring all input variables because all
// coefficients of them will be zero)
// The calcGrad param in case of true calculates the gradient in addition of the
// cost, and in case of false, only calculates the cost
func (nn *NeuralNet) CostFunction(lambda float64, calcGrad bool) (j float64, grad [][][]float64, err error) {
	if len(nn.Y) == 0 || len(nn.X) == 0 || len(nn.Theta) == 0 {
		err = fmt.Errorf("the lenght of the X, Y or Theta params are zero")
		return
	}

	if len(nn.Y) != len(nn.X) {
		err = fmt.Errorf(
			"The length of the X parameter doesn't corresponds with the length of the Y parameter")
		return
	}

	if len(nn.Theta[len(nn.Theta)-1]) != len(nn.Y[0]) {
		err = fmt.Errorf(
			"the length of the last theta layer should to correspond with the length of the expected results")
		return
	}

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
			delta[d] = mt.MultElems(mt.Mult(delta[d+1], nn.Theta[d+1]), addBias(mt.Apply(z[d], sigmoidGradient)))
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

// GetPerformance Returns the performance of the neural network with the current
// set of samples. The performance is calculated as: matches / total_samples
func (nn *NeuralNet) GetPerformance(verbose bool) (cost float64, performance float64) {
	matches := 0.0
	for i := 0; i < len(nn.X); i++ {
		match := true
		prediction := nn.Hipotesis(nn.X[i])

		for i := 0; i < len(prediction); i++ {
			if prediction[i] > 0.5 {
				prediction[i] = 1
			} else {
				prediction[i] = 0
			}
		}

	checkHip:
		for h := 0; h < len(prediction); h++ {
			if nn.Y[i][h] != prediction[h] {
				match = false
				break checkHip
			}
		}

		if match {
			matches++
		}
	}

	cost, _, _ = nn.CostFunction(0, false)
	performance = matches / float64(len(nn.Y))

	return
}

// Hipotesis returns the hipotesis calculation for the sample "x" using the
// thetas of nn.Theta
func (nn *NeuralNet) Hipotesis(x []float64) (result []float64) {
	aux := [][]float64{x}

	for _, theta := range nn.Theta {
		aux = mt.Apply(mt.Mult(addBias(aux), mt.Trans(theta)), sigmoid)
	}

	return aux[0]
}

// InitializeThetas Random inizialization of the thetas to break the simetry.
// The slice "layerSizes" will contain on each element, the size of the layer to
// be initialized, the first layer is the input one, and last layer will
// correspond to the output layer
func (nn *NeuralNet) InitializeThetas(layerSizes []int) {
	rand.Seed(int64(time.Now().Nanosecond()))
	epsilon := math.Sqrt(6) / math.Sqrt(float64(layerSizes[0]+layerSizes[len(layerSizes)-1]))

	nn.Theta = make([][][]float64, len(layerSizes)-1)

	for l := 1; l < len(layerSizes); l++ {
		nn.Theta[l-1] = make([][]float64, layerSizes[l])
		for n := 0; n < layerSizes[l]; n++ {
			nn.Theta[l-1][n] = make([]float64, layerSizes[l-1]+1)
			for i := 0; i < layerSizes[l-1]+1; i++ {
				if rand.Float64() > 0.5 {
					nn.Theta[l-1][n][i] = (rand.Float64() * epsilon)
				} else {
					nn.Theta[l-1][n][i] = 0 - (rand.Float64() * epsilon)
				}
			}
		}
	}

	return
}

// MinimizeCost This metod splits the samples contained in the NeuralNet instance
// in three sets (60%, 20%, 20%): training, cross validation and test. In order
// to calculate the optimal theta, after try with different lambda values on the
// training set and compare the performance obtained with the cross validation
// set to get the lambda with a better performance in the cross validation set.
// After calculate the best lambda, merges the training and cross validation
// sets and trains the neural network with the 80% of the samples.
// The data can be shuffled in order to obtain a better distribution before
// divide it in groups
func (nn *NeuralNet) MinimizeCost(maxIters int, suffleData bool, verbose bool) (finalCost float64, performance float64, trainingData *NeuralNet, testData *NeuralNet) {

	lambdas := []float64{0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300}

	if suffleData {
		nn = nn.shuffle()
	}

	// Get the 60% of the nn as training nn, 20% as cross validation, and
	// the remaining 20% as test nn
	training := int64(float64(len(nn.X)) * 0.6)
	cv := int64(float64(len(nn.X)) * 0.8)

	trainingData = &NeuralNet{
		X:     nn.X[:training],
		Y:     nn.Y[:training],
		Theta: nn.Theta,
	}

	cvData := &NeuralNet{
		X:     nn.X[training:cv],
		Y:     nn.Y[training:cv],
		Theta: nn.Theta,
	}
	testData = &NeuralNet{
		X:     nn.X[cv:],
		Y:     nn.Y[cv:],
		Theta: nn.Theta,
	}

	// Launch a process for each lambda in order to obtain the one with best
	// performance
	bestJ := math.Inf(1)
	bestLambda := 0.0
	initTheta := copyTheta(trainingData.Theta)

	for _, posLambda := range lambdas {
		if verbose {
			fmt.Println("Checking Lambda:", posLambda)
		}
		trainingData.Theta = copyTheta(initTheta)
		Fmincg(trainingData, posLambda, 3, verbose)
		cvData.Theta = trainingData.Theta

		j, _, _ := cvData.CostFunction(posLambda, false)

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
	Fmincg(trainingData, bestLambda, maxIters, verbose)

	testData.Theta = trainingData.Theta
	nn.Theta = trainingData.Theta

	finalCost, performance = testData.GetPerformance(verbose)

	return
}

// NewNeuralNetFromCsv Loads the informaton contained in the specified file
// paths and returns a NeuralNet instance.
// Each input file should contain a row by sample, and the values separated by a
// single space.
// To load the thetas specify on thetaSrc the file paths that contains each of
// the layer values. The order of this paths will represent the order of the
// layers.
// In case of need only to load the theta paramateres, specify a empty string on
// the xSrc and ySrc parameters.
func NewNeuralNetFromCsv(xSrc string, ySrc string, thetaSrc []string) (result *NeuralNet) {
	result = new(NeuralNet)

	if xSrc != "" {
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
	}

	if ySrc != "" {
		// Parse the Y params
		strInfo, err := ioutil.ReadFile(ySrc)
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
			result.Y = append(result.Y, values)
		}
	}

	// Parse the Theta params
	for _, thetaNSrc := range thetaSrc {
		strInfo, err := ioutil.ReadFile(thetaNSrc)
		if err != nil {
			panic(err)
		}

		trainingData := strings.Split(string(strInfo), "\n")
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

// SaveThetas Store all the current theta values of the instance in the
// "targetDir" directory.
// This method will create a file for each layer of theta called theta_X.txt
// where X is the layer contained on the file.
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

// addBias Returns a copy of the "m" two dim slice with a one added at the
// beginning of each row
func addBias(m [][]float64) (result [][]float64) {
	result = make([][]float64, len(m))

	for i := 0; i < len(m); i++ {
		result[i] = append([]float64{1}, m[i]...)
	}

	return
}

// copyTheta Returns a copy of the "theta" two dim slice allocated in a separate
// memory space
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

func (nn *NeuralNet) getTheta() [][][]float64 {
	return nn.Theta
}

// removeBias Returns a copy of the given two dim slice without the firs element
// of each row
func removeBias(x [][]float64) (result [][]float64) {
	result = make([][]float64, len(x))
	for i := 0; i < len(x); i++ {
		result[i] = append([]float64{0}, x[i][1:]...)
	}

	return
}

// rollThetasGrad returns a 1 x n matrix with the thetas concatenated
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

func (nn *NeuralNet) setTheta(t [][][]float64) {
	nn.Theta = t
}

// shuffle redistribute randomly all the X and Y rows of the instance
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
		shuffledData.Y[i] = dest[i][len(dest[i])-len(nn.Y[0]):]
		shuffledData.X[i] = dest[i][:len(dest[i])-len(nn.Y[0])]
	}

	shuffledData.Theta = nn.Theta

	return
}

func sigmoidGradient(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

// unrollThetasGrad Returns the 1 x n matrix as the multilayer theta way
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
