#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "relax.h"

/*
 *	The Kaczmarz iteration solves Ax = b for x
 *
 *	The algorithm iteratively projects the solution to a hyperplane
 *	defined by a (randomly) selected row.
 *
 *	This is achieved by computing the distance to the hyperplane by
 *	first evaluating it for x, and then moving x toward the hyperplane,
 *	in the normal direction of the hyperplane, by that distance.
 *
 *	The algorithm will operate on under- and overdetermined systems,
 *	as the principle is sound regardless of how many equations there
 *	are in the system.
 *
 *	Unfortunately the algorithm will not converge to a least squares
 *	solution for an overdetermined system: every iteration snaps it to
 *	one of the hyperplanes, it will not find a "compromise" in between
 *	the hyperplanes. [2] Describes a fix for this.
 *
 *	There has been renewed research interest in extending this algorithm
 *	[1][2][3], which is why it is included in this library.
 *
 *	[1] Thomas Strohmer, Roman Vershynin.
 *	    A Randomized Kaczmarz Algorithm with Exponential Convergence
 *
 *	[2] A. Zouzias and N. M. Freris.
 *	    Randomized extended Kaczmarz for solving least squares.
 *	    https://arxiv.org/pdf/1205.5770v3.pdf
 *
 *	[3] Anna Ma, Deanna Needell, Aaditya Ramdas.
 *	    Convergence properties of the randomized extended Gauss-Seidel
 *	    and Kaczmarz methods.
 *	    http://opt-ml.org/papers/OPT2015_paper_7.pdf
 */
int
relax_kacz(double *A, int m, int n, int stride, double *b, double *x0, int rowi)
{
	double rowres, fact;
	int ioff, j;

	if(rowi < 0 || rowi >= m)
		return -1;

	ioff = rowi * stride;
	rowres = relax_dot(A+ioff, 1, x0, 1, n) - b[rowi];
	fact = rowres / relax_dot(A+ioff, 1, A+ioff, 1, n);
	for(j = 0; j < n; j++)
		x0[j] -= fact*A[ioff+j];

	return 0;
}
