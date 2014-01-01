package ml

import (
	"github.com/alonsovidales/matrix"
	"math"
	"fmt"
)

type DataSet interface {
	CostFunction(lambda float64, calcGrad bool) (j float64, grad [][][]float64, err error)
	rollThetasGrad(x [][][]float64) [][]float64
	unrollThetasGrad(x [][]float64) [][][]float64
	setTheta(t [][][]float64)
	getTheta() [][][]float64
}

// Minimize a continuous differentialble multivariate function. Starting point
// is given by "X" (D by 1), and the function named in the string "f", must
// return a function value and a vector of partial derivatives. The Polack-
// Ribiere flavour of conjugate gradients is used to compute search directions,
// and a line search using quadratic and cubic polynomial approximations and the
// Wolfe-Powell stopping criteria is used together with the slope ratio method
// for guessing initial step sizes. Additionally a bunch of checks are made to
// make sure that exploration is taking place and that extrapolation will not
// be unboundedly large. The "length" gives the length of the run: if it is
// positive, it gives the maximum number of line searches, if negative its
// absolute gives the maximum allowed number of function evaluations. You can
// (optionally) give "length" a second component, which will indicate the
// reduction in function value to be expected in the first line-search (defaults
// to 1.0). The function returns when either its length is up, or if no further
// progress can be made (ie, we are at a minimum, or so close that due to
// numerical problems, we cannot get any closer). If the function terminates
// within a few iterations, it could be an indication that the function value
// and derivatives are not consistent (ie, there may be a bug in the
// implementation of your "f" function). The function returns the found
// solution "X", a vector of function values "fx" indicating the progress made
// and "i" the number of iterations (line searches or function evaluations,
// depending on the sign of "length") used.
//
// Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
//
//
// (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
// 
// Permission is granted for anyone to copy, use, or modify these
// programs and accompanying documents for purposes of research or
// education, provided this copyright notice is retained, and note is
// made of any changes that have been made.
// 
// These programs and documents are distributed without any warranty,
// express or implied.  As the programs were written for research
// purposes only, they have not been tested to the degree that would be
// advisable in any important application.  All use of these programs is
// entirely at the user's own risk.

func Fmincg(nn DataSet, lambda float64, length int, verbose bool) (fx[]float64, i int) {
	rho := 0.01  // a bunch of constants for line searches
	sig := 0.5   // RHO and SIG are the constants in the Wolfe-Powell conditions
	int := 0.1   // don't reevaluate within 0.1 of the limit of the current bracket
	ext := 3.0   // extrapolate maximum 3 times the current bracket
	max := 20    // max 20 function evaluations per line search
	ratio := 100.0 // maximum allowed slope ratio
	red := 1.0
	fx = []float64{}

	i = 0             // zero the run length counter
	lsFailed := false // no previous line search has failed

	f1, df1Tmp, _ := nn.CostFunction(lambda, true) // get function value and gradient
	df1 := nn.rollThetasGrad(df1Tmp)
	bestTheta := nn.getTheta()
	minCost := f1

	s := mt.Apply(df1, neg) // search direction is steepest
	d1 := mt.Mult(mt.Apply(s, neg), mt.Trans(s))[0][0] // this is the slope
	z1 := red / (float64(1) - d1) // initial step is red/(|s|+1)

	mainLoop: for i := 0; i < length; i++ {
		var z2 float64

		x0 := nn.rollThetasGrad(nn.getTheta()) // make a copy of current values
		f0 := f1
		df0 := mt.Copy(df1)
		x := mt.Sum(x0, mt.MultBy(s, z1)) // begin line search

		nn.setTheta(nn.unrollThetasGrad(x))
		f2, df2Temp, _ := nn.CostFunction(lambda, true)
		df2 := nn.rollThetasGrad(df2Temp)
		d2 := mt.Mult(df2, mt.Trans(s))[0][0]

		if f2 < minCost {
			bestTheta = nn.getTheta()
			minCost = f2
		}

		// initialize point 3 equal to point 1
		f3 := f1
		d3 := d1
		z3 := -z1

		success := false
		limit := -1.0
		searchLoop: for iters := 0; iters < max * 4; iters++ {
			m := max
			for ((f2 > f1 + z1 * rho * d1) || (d2 > -sig * d1)) && m > 0 {
				limit = z1
				if f2 > f1 {
					z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3)
				} else {
					a := 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
					b := 3 * (f3 - f2) - z3 * (d3 + 2 * d2)
					z2 = (math.Sqrt(b * b - a * d2 * z3 * z3) - b) / a // numerical error possible - ok!
				}

				if z2 != z2 || z2 == math.Inf(1) {
					z2 = z3 / 2 // if we had a numerical problem then bisect
				}

				z2 = math.Max(math.Min(z2, int * z3), (1 - int) * z3) // don't accept too close to limits
				z1 += z2 // update the step
				x = mt.Sum(x, mt.MultBy(s, z2))
				nn.setTheta(nn.unrollThetasGrad(x))
				f2, df2Temp, _ = nn.CostFunction(lambda, true)
				df2 = nn.rollThetasGrad(df2Temp)
				if f2 < minCost {
					bestTheta = nn.getTheta()
					minCost = f2
				}

				m--
				d2 = mt.Mult(df2, mt.Trans(s))[0][0]
				z3 -= z2
			}

			switch true {
				case f2 > f1 + z1 * rho * d1 || d2 > neg(sig) * d1: // this is a failure
					break searchLoop
				case d2 > sig * d1:
					success = true
					break searchLoop
				case m == 0: // failure
					break searchLoop
			}

			// make cubic extrapolation
			a := 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
			b := 3 * (f3 - f2) - z3 * (d3 + 2 * d2)
			z2 = -d2 * z3 * z3 / (b + math.Sqrt(b * b - a * d2 * z3 * z3)) // num. error possible - ok!

			switch true {
				case z2 != z2 || z2 < 0 || z2 == math.Inf(1): // num prob or wrong sign?
					z2 = z1 * (ext - 1)
					if limit < -0.5 {
						z2 = z1 * (ext - 1) // the extrapolate the maximum amount
					} else {
						z2 = (limit - z1) / 2 // otherwise bisect
					}
				case  limit > -0.5 && z2 + z1 > limit:
					z2 = (limit - z1) / 2
				case limit < -0.5 && z2 + z1 > z1 * ext:
					z2 = z1 * (ext - 1)
				case z2 < -z3 * int:
					z2 = -z3 * int
				case limit > -0.5 && z2 < (limit - z1) * (1 - int):
					z2 = (limit - z1) * (1 - int)
			}

			// set point 3 equal to point 2
			f3 = f2
			d3 = d2
			z3 = -z2
			z1 += z2
			x = mt.Sum(x, mt.MultBy(s, z2))
			nn.setTheta(nn.unrollThetasGrad(x))
			f2, df2Temp, _ = nn.CostFunction(lambda, true)
			if f2 < minCost {
				bestTheta = nn.getTheta()
				minCost = f2
			}
			df2 = nn.rollThetasGrad(df2Temp)

			m--
			d2 = mt.Mult(df2, mt.Trans(s))[0][0]
		}

		if success {
			f1 = f2
			fx = append(fx, f1)
			if verbose {
				fmt.Printf("Iteration: %d | Cost: %f\n", i + 1, f1)
			}

			// Polack-Ribiere direction
			s = mt.Sub(mt.MultBy(s, (mt.Mult(df2, mt.Trans(df2))[0][0] - mt.Mult(df1, mt.Trans(df2))[0][0]) / mt.Mult(df1, mt.Trans(df1))[0][0]), df2)

			// swap derivatives
			tmp := df1
			df1 = df2
			df2 = tmp

			d2 = mt.Mult(df1, mt.Trans(s))[0][0]
			if d2 > 0 {
				s = mt.Apply(df1, neg)
				d2 = mt.Mult(mt.Apply(s, neg), mt.Trans(s))[0][0]
			}
			z1 = z1 * math.Min(ratio, d1 / d2)
			d1 = d2
			lsFailed = false
		} else {
			nn.setTheta(nn.unrollThetasGrad(x0))
			f1 = f0
			df1 = df0
			if lsFailed || i > length {
				break mainLoop
			}
			tmp := df1
			df1 = df2
			df2 = tmp
			s = mt.Apply(df1, neg)
			d1 = mt.Mult(mt.Apply(s, neg), mt.Trans(s))[0][0]
			z1 = red / (float64(1) - d1)
			lsFailed = true
		}
	}

	nn.setTheta(bestTheta)
	return
}
