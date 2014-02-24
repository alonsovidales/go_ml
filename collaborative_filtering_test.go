package ml

import (
	"testing"
	"fmt"
	//"sort"
)

/*type movie struct {
	Score float64
	Id int
}
type moviesSort struct {
	movies []*movie
}

func (movies moviesSort) Len() int {
	return len(movies.movies)
}
func (movies moviesSort) Swap(i, j int) {
	movies.movies[i], movies.movies[j] = movies.movies[j], movies.movies[i]
}
func (movies moviesSort) Less(i, j int) bool {
	return movies.movies[i].Score < movies.movies[j].Score
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

	cf.InitializeThetas(10)
	cf.CalcMeans()
	Fmincg(cf, 10, 100, true)

	cf.AddUser(map[int]float64{
		0: 4,
		97: 2,
		6: 3,
		11: 5,
		53: 4,
		63: 5,
		65: 3,
		68: 5,
		182: 4,
		225: 5,
		354: 5,
	})

	Fmincg(cf, 10, 10, true)
	cf.MakePredictions()
	preds := cf.GetPredictionsFor(len(cf.AvailableRatings[0]) - 1)

	if preds[49] < 7.5 {
		t.Error("Error, the movie 49 was scored with:", preds[49], "and the expected score should to be > 7.5")
	}
	if preds[312] < 7.5 {
		t.Error("Error, the movie 312 was scored with:", preds[312], "and the expected score should to be > 7.5")
	}
	if preds[173] < 7.5 {
		t.Error("Error, the movie 173 was scored with:", preds[173], "and the expected score should to be > 7.5")
	}
	if preds[317] < 7.5 {
		t.Error("Error, the movie 317 was scored with:", preds[317], "and the expected score should to be > 7.5")
	}

	/*movies := new(moviesSort)
	for i, pred := range(preds) {
		movies.movies = append(movies.movies, &movie{
			Score: pred,
			Id: i,
		})
	}
	sort.Sort(movies)
	for _, movie := range(movies.movies) {
		fmt.Println(movie.Id, movie.Score)
	}*/
}
