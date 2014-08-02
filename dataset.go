package ml

import (
	"github.com/alonsovidales/go_matrix"
)

// DataSet Interface to be implemented by the machine learning algorithms to be
// used by the Fmincg function in order to reduce the cost
type DataSet interface {
	// Returns the cost and gradients for the current thetas configuration
	CostFunction(lambda float64, calcGrad bool) (j float64, grad []*mt.CudaMatrix, err error)
	// Returns the thetas in a 1xn matrix
	rollThetasGradTo(x []*mt.CudaMatrix, to *mt.CudaMatrix) *mt.CudaMatrix
	// Returns the thetas rolled by the rollThetasGrad method as it original form
	setRolledThetas(x *mt.CudaMatrix)
	// Sets the Theta param after convert it to the corresponding internal data structure
	setTheta(t []*mt.CudaMatrix)
	// Returns the theta as a 3 dimensional slice
	getTheta() []*mt.CudaMatrix

	// Prepares all the internal values to calculate the gradient in the fastest way possible
	InitFmincg()
}
