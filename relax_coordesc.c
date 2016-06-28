#include <stddef.h>
#include <math.h>
#include "relax.h"

/*
 *	The Gauss-Seidel iteration solves Ax = b for x
 *
 *	Instead of solving Ax = b directly, we'll reformulate the problem
 *	as finding the minimum of 0.5*xᵀAx - xᵀb instead.
 *
 *	The algorithm (randomly) picks a coordinate c and adjusts x_c
 * 	to minimize the residual.
 *
 *	The method is also known as coordinate descent, and is known to
 *	converge for Symmetric Positive (Semi-)Definite as well as
 *	diagonally dominant matrices
 *
 *
 *	There has been a recent development of the idea, extending this
 *	algorithm to work for both, under- and overdetermined systems.
 *
 *	[3] Anna Ma, Deanna Needell, Aaditya Ramdas.
 *	    Convergence properties of the randomized extended Gauss-Seidel
 *	    and Kaczmarz methods.
 *	    http://opt-ml.org/papers/OPT2015_paper_7.pdf
 */
double
relax_coordesc(double *A, int m, int n, int stride, double *b, double *x0, double *res)
{
	double sigma, rowres, maxres;
	int i, j, irow;

	maxres = 0.0;
	irow = 0;
	for(i = 0; i < m; i++){

		sigma = 0.0;
		rowres = 0.0;
		for(j = 0; j < n; j++){
			double tmp;
			tmp = A[irow+j] * x0[j];
			rowres += tmp;
			if(j != i)
				sigma += tmp;
		}

		x0[i] = (b[i] - sigma) / A[irow+i];

		rowres = b[i] - rowres;
		rowres = fabs(rowres);
		maxres = rowres > maxres ? rowres : maxres;
		if(res != NULL)
			res[i] = rowres;

		irow = irow + stride;
	}
	return maxres;
}
