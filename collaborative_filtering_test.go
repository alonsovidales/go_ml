package ml

import (
	"testing"
	"fmt"
	//"sort"
)

type movie struct {
	Score float64
	Id int
}
type moviesSort struct {
	movies []*movie
}

/*func (movies moviesSort) Len() int {
	return len(movies.movies)
}
func (movies moviesSort) Swap(i, j int) {
	movies.movies[i], movies.movies[j] = movies.movies[j], movies.movies[i]
}
func (movies moviesSort) Less(i, j int) bool {
	return movies.movies[i].Score > movies.movies[j].Score
}*/

func TestCollFilteringCostFunc(t *testing.T) {
	fmt.Println("Testing Collavorative Fitlers Cost Function...")

	cf := &CollaborativeFilter {
		Ratings: [][]float64{
			[]float64{5, 4, 0, 0},
			[]float64{3, 0, 0, 0},
			[]float64{4, 0, 0, 0},
			[]float64{3, 0, 0, 0},
			[]float64{3, 0, 0, 0},
		},
		AvailableRatings: [][]float64 {
			[]float64{1, 1, 0, 0},
			[]float64{1, 0, 0, 0},
			[]float64{1, 0, 0, 0},
			[]float64{1, 0, 0, 0},
			[]float64{1, 0, 0, 0},
		},
		ItemsTheta: [][]float64{
			[]float64{1.048686, -0.400232, 1.194119},
			[]float64{0.780851, -0.385626, 0.521198},
			[]float64{0.641509, -0.547854, -0.083796},
			[]float64{0.453618, -0.800218, 0.680481},
			[]float64{0.937538, 0.106090, 0.361953},
		},
		Theta: [][]float64{
			[]float64{0.28544, -1.68427, 0.26294},
			[]float64{0.50501, -0.45465, 0.31746},
			[]float64{-0.43192, -0.47880, 0.84671},
			[]float64{0.72860, -0.27189, 0.32684},
		},
	}

	j, _, err := cf.CostFunction(0.0, false)
	if err != nil {
		panic(err)
	}

	if j != 22.224626092180294 {
		t.Error("Expected cost: 22.224626092180294 , but", j, "returned")
	}
}

func TestCollFiltering(t *testing.T) {
	fmt.Println("Testing Collavorative Fitlers...")
	cf, err := NewCollFilterFromCsv(
		"test_data/collaborative_filtering/votes.csv",
		"test_data/collaborative_filtering/available_votes.csv",
		"test_data/collaborative_filtering/x.csv",
		"test_data/collaborative_filtering/theta.csv",
	)

	if err != nil {
		t.Error("Error loading the info from test files:", err)
	}

	userClassif := make([]float64, len(cf.Ratings))
	userClassif[0] = 4
	userClassif[97] = 2
	userClassif[6] = 3
	userClassif[11] = 5
	userClassif[53] = 4
	userClassif[63] = 5
	userClassif[65] = 3
	userClassif[68] = 5
	userClassif[182] = 4
	userClassif[225] = 5
	userClassif[354] = 5

	for i := 0; i < len(cf.Ratings); i++ {
		cf.Ratings[i] = append(cf.Ratings[i], userClassif[i])
		if userClassif[i] == 0 {
			cf.AvailableRatings[i] = append(cf.AvailableRatings[i], 0.0)
		} else {
			cf.AvailableRatings[i] = append(cf.AvailableRatings[i], 1.0)
		}
	}

	cf.InitializeThetas(10)
	//cf.Ratings = cf.Normalize()
	cf.CalcMeans()
	Fmincg(cf, 10, 100, true)
	cf.MakePredictions()
	preds := cf.GetPredictionsFor(len(cf.AvailableRatings[0]) - 1)

	if preds[49] < 8 {
		t.Error("Error, the movie 49 was scored with:", preds[49], "and the expected score should to be > 8")
	}
	if preds[312] < 8 {
		t.Error("Error, the movie 312 was scored with:", preds[312], "and the expected score should to be > 8")
	}
	if preds[173] < 8 {
		t.Error("Error, the movie 173 was scored with:", preds[173], "and the expected score should to be > 8")
	}
	if preds[317] < 8 {
		t.Error("Error, the movie 317 was scored with:", preds[317], "and the expected score should to be > 8")
	}
}
