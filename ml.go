// Package ml provides some implementations of usefull machine learning
// algorithms for data mining and data analysis.
//
// The implemented algorithms are:
//    - Linear Regression
//    - Logistic Regression
//    - Neural Networks
//    - Collaborative Filtering
//
// Is implemented too the fmincg function in order to calculate the optimal
// theta configuration to reduce the cost value for all the implemented solutions.
//
// Author: Alonso Vidales <alonso.vidales@tras2.es>
//
// Use of this source code is governed by a BSD-style.
// These programs and documents are distributed without any warranty, express or
// implied. All use of these programs is entirely at the user's own risk.
//
package ml

// General purpose machine learning functions

import (
	"math"
)

// Returns all the values of the given matrix normalized, the formula applied to
// all the elements is: (Xn - Avg) / (max - min) If all the elements in the
// slice have the same values, or the slice is empty, the slice can't be
// normalized, then returns false in the valid parameter
func Normalize(values []float64) (norm []float64, valid bool) {
	avg := 0.0
	max := math.Inf(-1)
	min := math.Inf(1)
	math.Inf(1)
	for _, val := range values {
		avg += val
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}

	if max == min || len(values) == 0 {
		valid = false
		return
	}

	valid = true
	avg /= float64(len(values))
	for _, val := range values {
		norm = append(norm, (val-avg)/(max-min))
	}

	return
}

// This method calculates all the possible combinations of the features and
// returns them with the specified degree, for example, for a data.X with x1, x2
// and degree 2 will convert data.X to 1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2,
// (x1 * x2) ** 2
// Use this method with care in order to calculate the model who fits better with
// the problem
func MapFeatures(x [][]float64, degree int) (ret [][]float64) {
	ret = make([][]float64, len(x))
	elems := len(x[0])
	for i := 0; i < len(x); i++ {
		aux := make([]float64, len(x[i]))
		copy(aux, x[i])
		ret[i] = make([]float64, len(x[i]))
		copy(ret[i], x[i])

		for l := 2; l <= elems; l++ {
			x[i] = append(x[i], combinations(aux, l)...)
		}
	}

	ret = PrepareX(x, degree)

	return
}

// Retrns the x matrix with all the elements at the power of x, x-1, x-2, ... 1
// and adds at the being of each row a 1 in order to be used as bias value
// For example for a given matrix like:
//    3 4
//    5 8
// Prepared at the power of 2 (x = 2):
//    1 3  9 4 16
//    1 5 25 8 64
func PrepareX(x [][]float64, degree int) (newX [][]float64) {
	for _, values := range x {
		result := []float64{1}

		for _, value := range values {
			for calcDeg := 1; calcDeg <= degree; calcDeg++ {
				result = append(result, math.Pow(value, float64(calcDeg)))
			}
		}

		newX = append(newX, result)
	}

	return
}

// Returns the result of multiply all the elements contained on the slice
func multElems(elems []float64) (resilt float64) {
	resilt = 1
	for _, elem := range elems {
		resilt *= elem
	}

	return
}

// Returns a slice with all the possible combinations of lenght "r" of the
// elements contained in the slice "iterable"
func combinations(iterable []float64, r int) (results []float64) {
	pool := iterable
	n := len(pool)

	if r > n {
		return
	}

	indices := make([]int, r)
	for i := range indices {
		indices[i] = i
	}

	result := make([]float64, r)
	for i, el := range indices {
		result[i] = pool[el]
	}

	results = append(results, multElems(result))
	for {
		i := r - 1
		for ; i >= 0 && indices[i] == i+n-r; i -= 1 {
		}

		if i < 0 {
			return
		}

		indices[i] += 1
		for j := i + 1; j < r; j += 1 {
			indices[j] = indices[j-1] + 1
		}

		for ; i < len(indices); i += 1 {
			result[i] = pool[indices[i]]
		}
		results = append(results, multElems(result))
	}

	return
}

// Auxiliar functions to work with matrix elements

// Returns the negation of the given float
func neg(n float64) float64 {
	return -n
}

// Calculates the sigmoid funcion for logistic regression
func sigmoid(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, neg(z)))
}

// Returns one minus the given float
func oneMinus(x float64) float64 {
	return 1 - x
}

// Returns the number at the power of two
func powTwo(x float64) float64 {
	return x * x
}
