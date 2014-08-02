package ml

import (
	"io/ioutil"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

// K-Means algorithm implementation for clustering
// @see http://en.wikipedia.org/wiki/K-means_clustering
type KMeans struct {
	// The different elements to be classified in the Groups number of
	// groups
	X [][]float64
	// The number of grops to be created
	Groups int
	// The current position of the centroids
	Centroids [][]float64

	// Will contain to which group the element belong
	xCentroids []int
}

// CalculateReturnGroups Calculates the centroids for all the specified groups,
// and returns the element positions in the X array grouped by the groups where
// the element belongs
// Use the initCentroids param to specify if the centroids should be be
// randomly initilizated, or just use the current centroids in the Centroids
// property
func (km *KMeans) CalculateReturnGroups(initCentroids bool) (xByGroups map[int][]int) {
	if initCentroids {
		km.initCentroids()
	}

	movedCentroids := true
	for movedCentroids {
		km.assignCentroids()
		movedCentroids = km.moveCentroids()
	}

	xByGroups = make(map[int][]int)
	for x := 0; x < len(km.X); x++ {
		if _, ok := xByGroups[km.xCentroids[x]]; ok {
			xByGroups[km.xCentroids[x]] = append(xByGroups[km.xCentroids[x]], x)
		} else {
			xByGroups[km.xCentroids[x]] = []int{x}
		}
	}

	return
}

// NewKmeansFromCsv Loads all the elements contained in the xSrc file, and
// stores them in the X property of a new object that is returned by the func.
// The format of this file should to be:
// X1_1 X1_2 X1_3 ... X1_n
// X2_1 X2_2 X2_3 ... X2_n
// ...
// Xn_1 Xn_2 Xn_3 ... Xn_n
func NewKmeansFromCsv(xSrc string) (result *KMeans) {
	result = new(KMeans)

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

	return
}

// initCentroids Random initialization of the centroids, the values for each
// centroid are chosen between the range of values for each feature in the X
// property
// IMPORTANT The X property should to be populated before call to this method
func (km *KMeans) initCentroids() {
	rand.Seed(int64(time.Now().Nanosecond()))
	km.Centroids = make([][]float64, km.Groups)

	for i := 0; i < km.Groups; i++ {
		km.Centroids[i] = make([]float64, len(km.X[0]))
		for f := 0; f < len(km.X[0]); f++ {
			max := math.Inf(-1)
			min := math.Inf(1)
			for x := 0; x < len(km.X); x++ {
				if max < km.X[x][f] {
					max = km.X[x][f]
				}
				if min > km.X[x][f] {
					min = km.X[x][f]
				}
			}
			km.Centroids[i][f] = ((max - min) * rand.Float64()) + min
		}
	}
}

// assignCentroids Calculates the distance between each point and each
// centroid, and assigns the closest centroid to each point
func (km *KMeans) assignCentroids() {
	km.xCentroids = make([]int, len(km.X))

	for i := 0; i < len(km.X); i++ {
		minDist := math.Inf(1)
		for c := 0; c < km.Groups; c++ {
			dist := 0.0
			for f := 0; f < len(km.X[0]); f++ {
				dist += math.Pow(km.Centroids[c][f]-km.X[i][f], 2)
			}

			dist /= float64(len(km.X[0]))
			if dist < minDist {
				km.xCentroids[i] = c
				minDist = dist
			}
		}
	}
}

// moveCentroids Recalculates the position of the centroids according to the
// set of elements assigned to this centroid
func (km *KMeans) moveCentroids() bool {
	movedCentroids := false
	newCentroids := make([][]float64, km.Groups)
	elemsByCentroid := make([]int, km.Groups)

	for i := 0; i < km.Groups; i++ {
		newCentroids[i] = make([]float64, len(km.X[0]))
	}

	for x := 0; x < len(km.X); x++ {
		for f := 0; f < len(km.X[0]); f++ {
			newCentroids[km.xCentroids[x]][f] += km.X[x][f]
		}
		elemsByCentroid[km.xCentroids[x]] += 1
	}

	for c := 0; c < km.Groups; c++ {
		for f := 0; f < len(km.X[0]); f++ {
			if elemsByCentroid[c] > 0 {
				newCentroids[c][f] /= float64(elemsByCentroid[c])
				if newCentroids[c][f] != km.Centroids[c][f] {
					movedCentroids = true
				}
			} else {
				newCentroids[c] = km.Centroids[c]
			}
		}
	}
	km.Centroids = newCentroids

	return movedCentroids
}
