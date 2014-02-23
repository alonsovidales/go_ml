package ml

import (
	"github.com/alonsovidales/go_matrix"
	"io/ioutil"
	"strings"
	"strconv"
	"math/rand"
	"time"
)

type CollaborativeFilter struct {
	// User ratios by item (rows), and user (cols)
	Ratings [][]float64
	// Matrix for classified or not items by user, use 0.0 for unclissified, 1.0 for classified
	AvailableRatings [][]float64
	// Martrix with items and features
	ItemsTheta [][]float64
	Theta [][]float64
	Means []float64
	Features int
	Predictions [][]float64
}

func (cf *CollaborativeFilter) GetPredictionsFor(userPos int) (preds []float64) {
	preds = make([]float64, len(cf.Ratings))
	for i, pred := range(cf.Predictions) {
		preds[i] = pred[userPos] + cf.Means[i]
	}

	return
}

func (cf *CollaborativeFilter) MakePredictions() {
	cf.Predictions = mt.MultTrans(cf.ItemsTheta, cf.Theta)
}

func (cf *CollaborativeFilter) AddUser(votes map[int]float64) {
	for i := 0; i < len(cf.Ratings); i++ {
		if score, ok := votes[i]; ok {
			cf.Ratings[i] = append(cf.Ratings[i], score)
			cf.AvailableRatings[i] = append(cf.AvailableRatings[i], 1.0)
		} else {
			cf.Ratings[i] = append(cf.Ratings[i], 0)
			cf.AvailableRatings[i] = append(cf.AvailableRatings[i], 0.0)
		}
	}

	rand.Seed(int64(time.Now().Nanosecond()))
	cf.Theta = append(cf.Theta, make([]float64, cf.Features))
	for i := 0; i < cf.Features; i++ {
		if rand.Float64() > 0.5 {
			cf.Theta[len(cf.Theta) - 1][i] = rand.Float64()
		} else {
			cf.Theta[len(cf.Theta) - 1][i] = 0 - rand.Float64()
		}
	}
}

func (cf *CollaborativeFilter) CalcMeans() () {
	cf.Means = make([]float64, len(cf.Ratings))
	width := len(cf.Ratings[0])
	for i := 0; i < len(cf.Ratings); i++ {
		scores := 0.0
		for j := 0; j < width; j++ {
			if cf.AvailableRatings[i][j] == 1 {
				cf.Means[i] += cf.Ratings[i][j]
				scores += 1.0
			}
		}
		cf.Means[i] /= scores
	}
}

func (cf *CollaborativeFilter) Normalize() (normRatings [][]float64) {
	if len(cf.Means) == 0 {
		cf.CalcMeans()
	}
	width := len(cf.Ratings[0])
	normRatings = make([][]float64, len(cf.Ratings))
	for i := 0; i < len(cf.Ratings); i++ {
		normRatings[i] = make([]float64, width)
		for j := 0; j < width; j++ {
			if cf.AvailableRatings[i][j] == 1 {
				normRatings[i][j] = cf.Ratings[i][j] - cf.Means[i]
			}
		}
	}

	return
}

func (cf *CollaborativeFilter) CostFunction(lambda float64, calcGrad bool) (j float64, grad [][][]float64, err error) {
	aux := mt.MultElems(mt.Sub(mt.MultTrans(cf.ItemsTheta, cf.Theta), cf.Ratings), cf.AvailableRatings)
	j = (mt.SumAll(mt.Apply(aux, powTwo)) / 2) + (lambda / 2 * mt.SumAll(mt.Apply(cf.Theta, powTwo))) + lambda / 2 * mt.SumAll(mt.Apply(cf.ItemsTheta, powTwo))
	if calcGrad {
		itemsGrad := mt.Sum(mt.Mult(aux, cf.Theta), mt.MultBy(cf.ItemsTheta, lambda))
		thetaGrad := mt.Sum(mt.Mult(mt.Trans(aux), cf.ItemsTheta), mt.MultBy(cf.Theta, lambda))

		grad = [][][]float64{
			itemsGrad,
			thetaGrad,
		}
	}

	return
}

func (cf *CollaborativeFilter) rollThetasGrad(x [][][]float64) [][]float64 {
	values := make([]float64, len(cf.ItemsTheta) * len(cf.ItemsTheta[0]) + len(cf.Theta) * len(cf.Theta[0]))
	for i := 0; i < len(cf.ItemsTheta); i++ {
		for j := 0; j < len(cf.ItemsTheta[0]); j++ {
			values[(i * len(cf.ItemsTheta[0])) + j] = x[0][i][j]
		}
	}

	for i := 0; i < len(cf.Theta); i++ {
		for j := 0; j < len(cf.Theta[0]); j++ {
			values[(len(cf.ItemsTheta) * len(cf.ItemsTheta[0])) + (i * len(cf.Theta[0])) + j] = x[1][i][j]
		}
	}

	return [][]float64{
		values,
	}
}

func (cf *CollaborativeFilter) unrollThetasGrad(x [][]float64) (result [][][]float64) {
	result = make([][][]float64, 2)
	result[0] = make([][]float64, len(cf.ItemsTheta))
	for i := 0; i < len(cf.ItemsTheta); i++ {
		result[0][i] = make([]float64, len(cf.ItemsTheta[0]))
		for j := 0; j < len(cf.ItemsTheta[0]); j++ {
			result[0][i][j] = x[0][(i * len(cf.ItemsTheta[0])) + j]
		}
	}

	result[1] = make([][]float64, len(cf.Theta))
	for i := 0; i < len(cf.Theta); i++ {
		result[1][i] = make([]float64, len(cf.Theta[0]))
		for j := 0; j < len(cf.Theta[0]); j++ {
			result[1][i][j] = x[0][(len(cf.ItemsTheta) * len(cf.ItemsTheta[0])) + (i * len(cf.Theta[0])) + j]
		}
	}

	return
}

func (cf *CollaborativeFilter) setTheta(t [][][]float64) {
	cf.ItemsTheta = t[0]
	cf.Theta = t[1]
}

func (cf *CollaborativeFilter) getTheta() [][][]float64 {
	return [][][]float64{
		cf.ItemsTheta,
		cf.Theta,
	}
}

func (cf *CollaborativeFilter) InitializeThetas(features int) {
	cf.Features = features
	rand.Seed(int64(time.Now().Nanosecond()))

	cf.ItemsTheta = make([][]float64, len(cf.Ratings))
	for j := 0; j < len(cf.Ratings); j++ {
		cf.ItemsTheta[j] = make([]float64, features)
		for i := 0; i < features; i++ {
			if rand.Float64() > 0.5 {
				cf.ItemsTheta[j][i] = rand.Float64()
			} else {
				cf.ItemsTheta[j][i] = 0 - rand.Float64()
			}
		}
	}

	cf.Theta = make([][]float64, len(cf.Ratings[0]))
	for j := 0; j < len(cf.Ratings[0]); j++ {
		cf.Theta[j] = make([]float64, features)
		for i := 0; i < features; i++ {
			if rand.Float64() > 0.5 {
				cf.Theta[j][i] = rand.Float64()
			} else {
				cf.Theta[j][i] = 0 - rand.Float64()
			}
		}
	}
}

func NewCollFilterFromCsv(ratingsSrc string, availableRatings string, itemsTheta string, theta string) (result *CollaborativeFilter, err error) {
	result = new(CollaborativeFilter)
	// Parse the Ratings params
	strInfo, err := ioutil.ReadFile(ratingsSrc)
	if err != nil {
		panic(err)
	}

	for _, line := range strings.Split(string(strInfo), "\n") {
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
		result.Ratings = append(result.Ratings, values)
	}


	// Parse the Ratings params
	strInfo, err = ioutil.ReadFile(availableRatings)
	if err != nil {
		panic(err)
	}
	for _, line := range strings.Split(string(strInfo), "\n") {
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
		result.AvailableRatings = append(result.AvailableRatings, values)
	}

	if itemsTheta != "" {
		// Parse the Ratings params
		strInfo, err = ioutil.ReadFile(itemsTheta)
		if err != nil {
			panic(err)
		}
		for _, line := range strings.Split(string(strInfo), "\n") {
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
			result.ItemsTheta = append(result.ItemsTheta, values)
		}
	}

	if theta != "" {
		// Parse the Ratings params
		strInfo, err = ioutil.ReadFile(theta)
		if err != nil {
			panic(err)
		}
		for _, line := range strings.Split(string(strInfo), "\n") {
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
			result.Theta = append(result.Theta, values)
		}
	}

	return
}
