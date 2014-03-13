package ml

import (
	"fmt"
	"testing"
)

func TestMultivariateGaussianDistreibution(t *testing.T) {
	fmt.Println("Testing Multivariate Gaussian Distribution implementation...")
	gd := MultVarGaussianDistLoadFromFile(
		"test_data/multivarite_gaussian_normal_distribution.dat")
	gd.CalculateMuSigma()
	probs := gd.GetProbability([][]float64{
		[]float64{13.0468, 14.7412},
		[]float64{13.4085, 13.7633},
		[]float64{14.1959, 15.8532},
		[]float64{14.9147, 16.1743},
	})

	expectedRes := []float64{
		0.06470823722117165,
		0.050304834321416796,
		0.07244977886910747,
		0.0503144083695188,
	}

	for i, res := range expectedRes {
		if res != probs[i] {
			t.Error("The expected value is:", res, "but the obtained was:", probs[i])
		}
	}
}
