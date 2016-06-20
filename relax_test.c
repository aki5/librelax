#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "relax.h"

int nloops = 10;
int jacobi;

int
main(int argc, char *argv[])
{
	double *A, *b, *x0, *x1, *res;
	double sigma, maxres;
	struct timeval tval;
	int i, j, irow;
	int m, n, stride;
	int loop;

	gettimeofday(&tval, NULL);
	srand48(tval.tv_sec ^ tval.tv_usec);
	m = 1000;
	n = 1000;
	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(m * sizeof b[0]);
	x0 = malloc(m * sizeof x0[0]);
	x1 = malloc(m * sizeof x1[0]);
	res = malloc(m * sizeof res[0]);

	if(argc > 1 && !strcmp(argv[1], "jacobi"))
		jacobi++;

	for(loop = 0; loop < nloops; loop++){
		irow = 0;
		for(i = 0; i < m; i++){
			sigma = 0.0;
			for(j = 0; j < n; j++){
				if(i != j){
					A[irow+j] = drand48();
					sigma += A[irow+j];
				}
			}
			// ensure matrix is somewhat diagonally dominant
			A[irow+i] = sigma + drand48();
			irow += stride;
		}

		for(i = 0; i < m; i++){
			b[i] = drand48();
			x0[i] = 0.0;
		}

		if(!jacobi)
			x1 = x0;
		for(i = 0; i < 100; i++){
			double *tmp;
			if(!jacobi)
				maxres = relax_step(A, m, n, stride, b, x0, x1, res, 0.9); // for gauss-seidel
			else
				maxres = relax_step(A, m, n, stride, b, x0, x1, res, 0.6); // for jacobi
			tmp = x0;
			x0 = x1;
			x1 = tmp;
			if(maxres < 1e-10)
				break;
		}
		printf("%d.%d: maxres %.20f\n", loop, i, maxres);
	}
	return 0;
}