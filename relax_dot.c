#include "relax.h"
/*
 *	relax_dot computes the dot product between the vectors a and b.
 *
 *	the stride parameters can be used to use a matrix column as a vector,
 *	but should be passed in as 1 for row vectors or when using a matrix row.
 */
double
relax_dot(double *a, int astride, double *b, int bstride, int n)
{
	double dot;
	int i, aoff, boff;

	aoff = 0;
	boff = 0;
	dot = 0.0;
	for(i = 0; i < n; i++){
		dot += a[aoff] * b[boff];
		aoff += astride;
		boff += bstride;
	}
	return dot;
}
