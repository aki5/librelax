#include "relax.h"
/*
 *	relax_ypax computes y = y + alpha*x
 *
 *	the stride parameters can be used to use a matrix column as a vector,
 *	but should be passed in as 1 for row vectors or when using a matrix row.
 */
void
relax_bypax(double beta, double *y, int ystride, double alpha, double *x, int xstride, int n)
{
	int i, xoff, yoff;

	xoff = 0;
	yoff = 0;
	for(i = 0; i < n; i++){
		y[yoff] = beta*y[yoff] + alpha*x[xoff];
		xoff += xstride;
		yoff += ystride;
	}
}
