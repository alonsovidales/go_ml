package ml

import (
	"testing"
	"fmt"
)

func TestCentroidsCalculation(t *testing.T) {
	fmt.Println("Testing K-means calculating centroids for the tests data...")
	nk := NewKmeansFromCsv("test_data/k_means.dat")
	nk.Groups = 3
	nk.Centroids = [][]float64{
		[]float64{3, 3},
		[]float64{6, 2},
		[]float64{8, 5},
	}

	groups := nk.CalculateReturnGroups(false)

	finalCentroids := [][]float64{
		[]float64{1.953994664859387, 5.025570059426876},
		[]float64{3.0436711927398132, 1.0154104079486552},
		[]float64{6.033667356017604, 3.0005251118352563},
	}

	for i := 0; i < len(finalCentroids); i++ {
		for f := 0; f < len(finalCentroids[0]); f++ {
			if finalCentroids[i][f] != nk.Centroids[i][f] {
				t.Error("Expectet centroid:", finalCentroids[i][f], "but:",  nk.Centroids[i][f], "obtained")
			}
		}
	}

	if len(groups[0]) != 98 {
		t.Error("Expectet length for the group 0: 98, obtained:", len(groups[0]))
	}
	if len(groups[1]) != 102 {
		t.Error("Expectet length for the group 0: 102, obtained:", len(groups[1]))
	}
	if len(groups[2]) != 100 {
		t.Error("Expectet length for the group 0: 100, obtained:", len(groups[2]))
	}
}
