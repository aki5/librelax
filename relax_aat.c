#include "relax.h"
/*
 *	relax_aat computes A*Aᵀ and stores the result in C
 *	A is MxN (has m rows and n columns)
 *	C is MxM (has m rows and m columns)
 */
void
relax_aat(double *A, int m, int n, int astride, double *C, int cstride)
{
	double sum;
	int i, j, k;
	int ioff, joff, cioff;

	ioff = 0;
	cioff = 0;
	for(i = 0; i < m; i++){
		joff = 0;
		for(j = 0; j < m; j++){
			sum = 0.0;
			for(k = 0; k < n; k++){
				// ik*kj, A[joff+k] is Aᵀ[koff+j]
				sum += A[ioff+k]*A[joff+k];
			}
			C[cioff+j] = sum;
			joff += astride;
		}
		ioff += astride;
		cioff += cstride;
	}
}
