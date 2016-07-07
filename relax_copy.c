#include "relax.h"

/*
 *	relax_copy copies the n-vector x to the n-vector y
 *
 *	as for other routines, the stride parameter is generally 1, but other values
 *	can be used to use non-consecutive linear patterns like matrix columns as vectors.
 */
void
relax_copy(double *y, int ystride, double *x, int xstride, int n)
{
	int i, xoff, yoff;

	xoff = 0;
	yoff = 0;
	for(i = 0; i < n; i++){
		y[yoff] = x[xoff];
		xoff += xstride;
		yoff += ystride;
	}
}
