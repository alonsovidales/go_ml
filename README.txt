PACKAGE DOCUMENTATION

package ml
    import "."

    Package ml provides some implementations of usefull machine learning
    algorithms for data mining and data analysis.

    The implemented algorithms are:

	- Linear Regression
	- Logistic Regression
	- Neural Networks
	- Collaborative Filtering

    Is implemented too the fmincg function in order to calculate the optimal
    theta configuration to reduce the cost value for all the implemented
    solutions.

    Author: Alonso Vidales <alonso.vidales@tras2.es>

    Use of this source code is governed by a BSD-style. These programs and
    documents are distributed without any warranty, express or implied. All
    use of these programs is entirely at the user's own risk.


FUNCTIONS

func Fmincg(nn DataSet, lambda float64, length int, verbose bool) (fx []float64, i int, err error)
    Minimize a continuous differentialble multivariate function. Starting
    point is given by the "Lambda" property (D by 1), and the method named
    "CostFunction", must return a function value and a vector of partial
    derivatives. The Polack- Ribiere flavour of conjugate gradients is used
    to compute search directions, and a line search using quadratic and
    cubic polynomial approximations and the Wolfe-Powell stopping criteria
    is used together with the slope ratio method for guessing initial step
    sizes. Additionally a bunch of checks are made to make sure that
    exploration is taking place and that extrapolation will not be
    unboundedly large. The "length" gives the length of the run: if it is
    positive, it gives the maximum number of line searches, if negative its
    absolute gives the maximum allowed number of function evaluations. The
    function returns when either its length is up, or if no further progress
    can be made (ie, we are at a minimum, or so close that due to numerical
    problems, we cannot get any closer). If the function terminates within a
    few iterations, it could be an indication that the function value and
    derivatives are not consistent (ie, there may be a bug in the
    implementation of your "f" function). The function returns "fx"
    indicating the progress made and "i" the number of iterations (line
    searches or function evaluations, depending on the sign of "length")
    used.

    Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
    Ported from Octave to Go by Alonso Vidales <alonso.vidales@tras2.es>

    (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen

    Permission is granted for anyone to copy, use, or modify these programs
    and accompanying documents for purposes of research or education,
    provided this copyright notice is retained, and note is made of any
    changes that have been made.

    These programs and documents are distributed without any warranty,
    express or implied. As the programs were written for research purposes
    only, they have not been tested to the degree that would be advisable in
    any important application. All use of these programs is entirely at the
    user's own risk.

func MapFeatures(x [][]float64, degree int) (ret [][]float64)
    This method calculates all the possible combinations of the features and
    returns them with the specified degree, for example, for a data.X with
    x1, x2 and degree 2 will convert data.X to 1, x1, x2, x1 * x2, x1 ** 2,
    x2 ** 2, (x1 * x2) ** 2 Use this method with care in order to calculate
    the model who fits better with the problem

func Normalize(values []float64) (norm []float64, valid bool)
    Returns all the values of the given matrix normalized, the formula
    applied to all the elements is: (Xn - Avg) / (max - min) If all the
    elements in the slice have the same values, or the slice is empty, the
    slice can't be normalized, then returns false in the valid parameter

func PrepareX(x [][]float64, degree int) (newX [][]float64)
    Retrns the x matrix with all the elements at the power of x, x-1, x-2,
    ... 1 and adds at the being of each row a 1 in order to be used as bias
    value For example for a given matrix like:

	3 4
	5 8

    Prepared at the power of 2 (x = 2):

	1 3  9 4 16
	1 5 25 8 64


TYPES

type CollaborativeFilter struct {
    // User ratios by item (rows), and user (cols)
    Ratings [][]float64
    // Matrix for classified or not items by user, use 0.0 for unclissified, 1.0 for classified
    AvailableRatings [][]float64
    // Martrix with items and features
    ItemsTheta [][]float64
    Theta      [][]float64
    // Used for mean normalization, will store the mean ratings for all the items
    Means       []float64
    Features    int
    Predictions [][]float64
}
    Collaborative filtering implementation, this algorithm is able to
    determine the items with a best fit for items not yet rated in a matrix
    of users and items calcifications:
    http://en.wikipedia.org/wiki/Collaborative_filtering


func NewCollFilterFromCsv(ratingsSrc string, availableRatings string, itemsTheta string, theta string) (result *CollaborativeFilter, err error)
    Loads the information from the CSV space separated files for the
    collaborative filter


func (cf *CollaborativeFilter) AddUser(votes map[int]float64)
    Adds a single user ratings to the user ratings matrix and prepares the
    theta parameters, to calculate the recommendations for this user

func (cf *CollaborativeFilter) CalcMeans()
    Calculate the means for all the items and store them

func (cf *CollaborativeFilter) CostFunction(lambda float64, calcGrad bool) (j float64, grad [][][]float64, err error)
    Cost function for the collaborative filter

func (cf *CollaborativeFilter) GetPredictionsFor(userPos int) (preds []float64)
    Returns the predictions for a single user in the given position

func (cf *CollaborativeFilter) InitializeThetas(features int)
    Random initialization of the thetas for the given features

func (cf *CollaborativeFilter) MakePredictions()
    Prepare the predictions for all the users

func (cf *CollaborativeFilter) Normalize() (normRatings [][]float64)
    Normalize the rating of the users, this method doesn't update the
    ratings in the objects, just returns them


type DataSet interface {
    // Returns the cost and gradients for the current thetas configuration
    CostFunction(lambda float64, calcGrad bool) (j float64, grad [][][]float64, err error)
    // contains filtered or unexported methods
}
    Interface to be implemented by the machine learning algorithms to be
    used by the Fmincg function in order to reduce the cost



type NeuralNet struct {
    // Training set of values for each feature, the first dimension are the test cases
    X [][]float64
    // The training set with values to be predicted
    Y [][]float64
    // 1st dim -> layer, 2nd dim -> neuron, 3rd dim theta
    Theta [][][]float64
}
    Neural network representation, the X and Y properties are to be used
    with training proposals


func NewNeuralNetFromCsv(xSrc string, ySrc string, thetaSrc []string) (result *NeuralNet)
    Loads the informaton contained in the specified file paths and returns a
    NeuralNet instance. Each input file should contain a row by sample, and
    the values separated by a single space. To load the thetas specify on
    thetaSrc the file paths that contains each of the layer values. The
    order of this paths will represent the order of the layers. In case of
    need only to load the theta paramateres, specify a empty string on the
    xSrc and ySrc parameters.


func (nn *NeuralNet) CostFunction(lambda float64, calcGrad bool) (j float64, grad [][][]float64, err error)
    Calcualtes the cost function for the training set stored in the X and Y
    properties of the instance, and with the theta configuration. The lambda
    parameter controls the degree of regularization (0 means
    no-regularization, infinity means ignoring all input variables because
    all coefficients of them will be zero) The calcGrad param in case of
    true calculates the gradient in addition of the cost, and in case of
    false, only calculates the cost

func (nn *NeuralNet) GetPerformance(verbose bool) (cost float64, performance float64)
    Returns the performance of the neural network with the current set of
    samples. The performance is calculated as: matches / total_samples

func (nn *NeuralNet) Hipotesis(x []float64) (result []float64)
    Returns the hipotesis calculation for the sample "x" using the thetas of
    nn.Theta

func (nn *NeuralNet) InitializeThetas(layerSizes []int)
    Random inizialization of the thetas to break the simetry. The slice
    "layerSizes" will contain on each element, the size of the layer to be
    initialized, the first layer is the input one, and last layer will
    correspond to the output layer

func (nn *NeuralNet) MinimizeCost(maxIters int, suffleData bool, verbose bool) (finalCost float64, performance float64, trainingData *NeuralNet, testData *NeuralNet)
    This metod splits the samples contained in the NeuralNet instance in
    three sets (60%, 20%, 20%): training, cross validation and test. In
    order to calculate the optimal theta, after try with different lambda
    values on the training set and compare the performance obtained with the
    cross validation set to get the lambda with a better performance in the
    cross validation set. After calculate the best lambda, merges the
    training and cross validation sets and trains the neural network with
    the 80% of the samples. The data can be shuffled in order to obtain a
    better distribution before divide it in groups

func (nn *NeuralNet) SaveThetas(targetDir string) (files []string)
    Store all the current theta values of the instance in the "targetDir"
    directory. This method will create a file for each layer of theta called
    theta_X.txt where X is the layer contained on the file.


type Regression struct {
    X [][]float64 // Training set of values for each feature, the first dimension are the test cases
    Y []float64   // The training set with values to be predicted
    // 1st dim -> layer, 2nd dim -> neuron, 3rd dim theta
    Theta     []float64
    LinearReg bool // true indicates that this is a linear regression problem, false a logistic regression one
}
    Linear and logistic regression structure


func LoadFile(filePath string) (data *Regression)
    Loads information from the local file located at filePath, and after
    parse it, returns the Regression ready to be used with all the
    information loaded The file format is:

	X11 X12 ... X1N Y1
	X21 X22 ... X2N Y2
	... ... ... ... ..
	XN1 XN2 ... XNN YN

    Note: Use a single space as separator


func (lr *Regression) CostFunction(lambda float64, calcGrad bool) (j float64, grad [][][]float64, err error)
    Calcualtes the cost function for the training set stored in the X and Y
    properties of the instance, and with the theta configuration. The lambda
    parameter controls the degree of regularization (0 means
    no-regularization, infinity means ignoring all input variables because
    all coefficients of them will be zero) The calcGrad param in case of
    true calculates the gradient in addition of the cost, and in case of
    false, only calculates the cost

func (lr *Regression) InitializeTheta()
    Initialize the Theta property to an array of zeros with the lenght of
    the number of features on the X property

func (data *Regression) LinearHipotesis(x []float64) (r float64)

func (data *Regression) LogisticHipotesis(x []float64) (r float64)
    Returns the hipotesis result for the thetas in the instance and the
    specified parameters

func (data *Regression) MinimizeCost(maxIters int, suffleData bool, verbose bool) (finalCost float64, trainingData *Regression, lambda float64, testData *Regression)
    This metod splits the given data in three sets: training, cross
    validation, test. In order to calculate the optimal theta, tries with
    different possibilities and the training data, and check the best match
    with the cross validations, after obtain the best lambda, check the
    perfomand against the test set of data



SUBDIRECTORIES

	src
	test_data

