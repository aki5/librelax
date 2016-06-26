#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "relax.h"

/*
 *	the kaczmarz iteration solves Ax = b for x by adjusting the guess x0 so that
 *	the residual corresponding with that row becomes zero.
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

	
	//fprintf(stderr, "relax_kacz: row %d: rowres %f result %f\n", rowi, rowres, relax_dot(A+arow, 1, x0, 1, n) - b[rowi]);

	return 0;
}
