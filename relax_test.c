#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <fenv.h>
#include <float.h>
#include "relax.h"

int nloops = 10;
int jacobi;

void
build_system(double *A, int m, int n, int stride, double *b)
{
	double sigma;

	int i, j, irow;

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
	}
}

void
init_guess(double *x0, int n)
{
	int j;
	for(j = 0; j < n; j++)
		x0[j] = 0.0;
}

void
iterate_ssor(double *A, int m, int n, int stride, double *b, double *x0, double *x1, double w)
{
	double maxres;
	int i;

	for(i = 0; i < 100; i++){
		double *tmp;

		feclearexcept(FE_ALL_EXCEPT);
		maxres = relax_sor(A, m, n, stride, b, x0, x1, NULL, w); // for gauss-seidel
		if(fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT)){
			fprintf(stderr, "iteration %d: maxres %.20f\n", i, maxres);
			fprintf(stderr,
				"floating point exception: %s%s%s%s%s\n",
				fetestexcept(FE_DIVBYZERO) ? " FE_DIVBYZERO" : "",
				fetestexcept(FE_INEXACT) ? " FE_INEXACT" : "",
				fetestexcept(FE_INVALID) ? " FE_INVALID" : "",
				fetestexcept(FE_OVERFLOW) ? " FE_OVERFLOW" : "",
				fetestexcept(FE_UNDERFLOW) ? " FE_UNDERFLOW" : ""
			);
			exit(1);
		}

		tmp = x0;
		x0 = x1;
		x1 = tmp;
		if(maxres < 1e-10)
			break;
	}
	printf("%d: maxres %.20f\n", i, maxres);
}

void
square_test(int input_n)
{
	double *A, *b, *x0, *x1;
	int m, n, stride;
	int loop;

	m = input_n;
	n = input_n;
	fprintf(stderr, "square_test %dx%d\n", m, n);

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(n * sizeof b[0]);
	x0 = malloc(n * sizeof x0[0]);
	if(jacobi)
		x1 = malloc(n * sizeof x1[0]);
	else
		x1 = x0;

	for(loop = 0; loop < nloops; loop++){
		build_system(A, m, n, stride, b);
		init_guess(x0, n);
		iterate_ssor(A, m, n, stride, b, x0, x1, jacobi ? 0.6 : 0.9);
	}

	free(A);
	free(b);
	free(x0);
	if(jacobi)
		free(x1);
}

void
lsq_test(int input_n)
{
	double *A, *b, *x0, *x1;
	double *C, *c;
	int m, n, stride;
	int loop;

	m = input_n + lrand48()%input_n;
	n = input_n;
	fprintf(stderr, "lsq_test %dx%d\n", m, n);

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(n * sizeof b[0]);
	x0 = malloc(n * sizeof x0[0]);
	if(jacobi)
		x1 = malloc(n * sizeof x1[0]);
	else
		x1 = x0;

	C = malloc(n * n * sizeof C[0]);
	c = malloc(n * sizeof c[0]);

	for(loop = 0; loop < nloops; loop++){
		build_system(A, m, n, stride, b);
		relax_ata(A, m, n, stride, C, n);
		relax_atb(A, m, n, stride, b, c);
		init_guess(x0, n);
		iterate_ssor(C, n, n, stride, c, x0, x1, jacobi ? 0.4 : 0.9);

	}

	free(A);
	free(b);
	free(x0);
	if(jacobi)
		free(x1);
}

int
main(int argc, char *argv[])
{
	struct timeval tval;

	if(argc > 1 && !strcmp(argv[1], "jacobi"))
		jacobi++;

	gettimeofday(&tval, NULL);
	srand48(tval.tv_sec ^ tval.tv_usec);

	square_test(1000);
	lsq_test(500);

	return 0;
}