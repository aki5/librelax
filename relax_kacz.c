#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "relax.h"

/*
 *	The Kaczmarz iteration solves Ax = b for x
 *
 *	Mathematically this is orthogonally projecting the solution to a
 *	hyperplane defined by the selected row.
 *
 *	Technically that translates to adding a multiple of the selected row
 *	to the current guess, while selecting a multiplier that causes the
 *	row residual to become zero.
 *
 *	The convergence properties of this algorithm are poorly understood.
 *	It seems to perform poorly on random matrices and their normal forms,
 *	but compares with gauss-seidel on diagonally dominant matrices.
 */
int
relax_kacz(double *A, int m, int n, int stride, double *b, double *x0, int rowi)
{
	double rowres, fact;
	int i, arow;

	if(rowi < 0 || rowi >= m)
		return -1;

	arow = rowi * stride;
	rowres = relax_dot(A+arow, 1, x0, 1, n) - b[rowi];
	fact = rowres / relax_dot(A+arow, 1, A+arow, 1, n);
	for(i = 0; i < n; i++)
		x0[i] -= fact*A[arow+i];


	return 0;
}
