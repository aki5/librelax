#include "relax.h"
/*
 *	relax_ab multiplies b by A and stores the result in c
 *	A is MxN (has m rows and n columns)
 *	c is Nx1 (has n rows)
 */
void
relax_ab(double *A, int m, int n, int astride, double *b, double *c)
{
	double sum;
	int i, k;
	int ioff;

	ioff = 0;
	for(i = 0; i < m; i++){
		sum = 0.0;
		for(k = 0; k < n; k++)
			sum += A[ioff+k]*b[k];
		c[i] = sum;
		ioff += astride;
	}
}
