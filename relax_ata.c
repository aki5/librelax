#include "relax.h"
/*
 *	relax_aat computes Aᵀ*A and stores the result in C
 *	A is MxN (has m rows and n columns)
 *	C is NxN (has n rows and n columns)
 */
void
relax_ata(double *A, int m, int n, int astride, double *C, int cstride)
{
	double sum;
	int i, j, k;
	int ioff, koff;

	ioff = 0;
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			sum = 0.0;
			koff = 0;
			for(k = 0; k < m; k++){
				// ik*kj, Aᵀ[ioff+k] is A[koff+i]
				sum += A[koff+i]*A[koff+j];
				koff += astride;
			}
			C[ioff+j] = sum;
		}
		ioff += cstride;
	}
}
