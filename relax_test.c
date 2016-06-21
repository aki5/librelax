#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <fenv.h>
#include "relax.h"

int nloops = 10;
int jacobi;

int
main(int argc, char *argv[])
{
	double *A, *b, *x0, *x1;
	double sigma, maxres;
	struct timeval tval;
	int i, j, irow;
	int m, n, stride;
	int loop;

	if(argc > 1 && !strcmp(argv[1], "jacobi"))
		jacobi++;

	gettimeofday(&tval, NULL);
	srand48(tval.tv_sec ^ tval.tv_usec);

	feclearexcept(FE_ALL_EXCEPT);
	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);

	m = 1000;
	n = 1000;
	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(n * sizeof b[0]);
	x0 = malloc(n * sizeof x0[0]);
	if(jacobi)
		x1 = malloc(n * sizeof x1[0]);
	else
		x1 = x0;

	for(loop = 0; loop < nloops; loop++){
		irow = 0;
		for(i = 0; i < m; i++){
			sigma = 0.0;
			for(j = 0; j < n; j++){
				if(i != j){
					A[irow+j] = drand48();
					sigma += fabs(A[irow+j]);
				}
			}
			// ensure matrix is diagonally dominant, ie. sum of magnitudes of other entries is
			// less than or equal to the magnitude of the diagonal entry.
			if(i < n){
				A[irow+i] = sigma + drand48();
				irow += stride;
			}
		}

		for(j = 0; j < n; j++){
			b[j] = drand48();
			x0[j] = 0.0;
		}

		for(i = 0; i < 100; i++){
			double *tmp;
			if(!jacobi)
				maxres = relax_sor(A, m, n, stride, b, x0, x1, NULL, 0.9); // for gauss-seidel
			else
				maxres = relax_sor(A, m, n, stride, b, x0, x1, NULL, 0.6); // for jacobi
			tmp = x0;
			x0 = x1;
			x1 = tmp;
			if(maxres < 1e-13)
				break;
		}
		printf("%d.%d: maxres %.20f\n", loop, i, maxres);
	}
	return 0;
}