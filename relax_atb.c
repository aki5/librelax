#include "relax.h"
/*
 *	relax_atb multiplies b by transpose of A, store the result in c
 *	A is MxN (has m rows and n columns)
 *	c is Nx1 (has n rows)
 */
void
relax_atb(double *A, int m, int n, int astride, double *b, double *c)
{
	double sum;
	int i, j, k;
	int ioff, koff;

	ioff = 0;
	for(i = 0; i < n; i++){
		sum = 0.0;
		koff = 0;
		for(k = 0; k < m; k++){
			sum += A[koff+i]*b[k];
			koff += astride;
		}
		c[i] = sum;
	}
}
