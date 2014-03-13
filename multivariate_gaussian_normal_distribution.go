package ml

import (
	"io/ioutil"
	"math"
	"strconv"
	"strings"
)

// Anomaly detection implementation using Multivariate Gaussian Distribution:
// http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function
type MultGaussianDist struct {
	X      [][]float64 // Data to train the gussian distribution
	Sigma2 []float64
	Mu     []float64 // Medians for each feature
}

// To be called before GetProbability method in order to calculate the medians
// and sigmas params for all the training set
func (gd *MultGaussianDist) CalculateMuSigma() {
	gd.Mu = make([]float64, len(gd.X[0]))

	for _, x := range gd.X {
		for i, val := range x {
			gd.Mu[i] += val
		}
	}
	for i := 0; i < len(gd.X[0]); i++ {
		gd.Mu[i] /= float64(len(gd.X))
	}

	gd.Sigma2 = make([]float64, len(gd.X[0]))
	for _, x := range gd.X {
		for i, val := range x {
			gd.Sigma2[i] += math.Pow(val-gd.Mu[i], 2)
		}
	}
	for i := 0; i < len(gd.X[0]); i++ {
		gd.Sigma2[i] /= float64(len(gd.X))
	}
}

// Returns the probability of anomaly for each data, each row of data is a
// sample to study and each colum a featurea, determinate an epsilon and when
// p(x) < epsilon, you may have an anomaly, you can use SelectThreshold in
// in order to calculate the best epsilon
func (gd *MultGaussianDist) GetProbability(data [][]float64) (p []float64) {
	d := make([][]float64, len(data))
	for i := 0; i < len(data); i++ {
		d[i] = make([]float64, len(data[i]))
		for c := 0; c < len(data[0]); c++ {
			d[i][c] = data[i][c] - gd.Mu[c]
		}
	}

	detSig := 1.0
	for i := 0; i < len(gd.Sigma2); i++ {
		detSig *= gd.Sigma2[i]
	}

	base := math.Pow(2*math.Pi, -(float64(len(d[0])))/2) * math.Pow(detSig, -0.5)
	p = make([]float64, len(d))
	for i := 0; i < len(d); i++ {
		for c := 0; c < len(d[0]); c++ {
			p[i] += -0.5 * d[i][c] * d[i][c] * (1 / gd.Sigma2[c])
		}
		p[i] = base * math.Pow(math.E, p[i])
	}
	return
}

// Creates a MultGaussianDist object from the content of a CSV file space
// sepparate where each line is a sample and each column a feature
func MultVarGaussianDistLoadFromFile(filePath string) (gd *MultGaussianDist) {
	strInfo, err := ioutil.ReadFile(filePath)
	if err != nil {
		panic(err)
	}

	trainingData := strings.Split(string(strInfo), "\n")
	gd = &MultGaussianDist{
		X: make([][]float64, len(trainingData)-1),
	}
	for i, line := range trainingData {
		if line == "" {
			break
		}

		parts := strings.Split(line, " ")
		gd.X[i] = make([]float64, len(parts))
		for c, value := range strings.Split(line, " ") {
			floatVal, err := strconv.ParseFloat(value, 64)
			if err != nil {
				panic(err)
			}
			gd.X[i][c] = floatVal
		}
	}

	return
}
