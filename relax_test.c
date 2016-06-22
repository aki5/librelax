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
dump_system(double *A, int m, int n, int stride, double *b)
{
	int i, j;
	for(i = 0; i < m; i++){
		for(j = 0; j < n; j++){
			printf(" %6.2f", A[i*stride+j]);
		}
		printf(" = %6.2f", b[i]);
		printf("\n");
	}
	printf("\n");
}

void
build_system(double *A, int m, int n, int stride, double *b, int diagdom)
{
	double sigma;

	int i, j, irow;

	irow = 0;
	for(i = 0; i < m; i++){
		sigma = 0.0;
		for(j = 0; j < n; j++){
			A[irow+j] = drand48();
			if(i != j)
				sigma += fabs(A[irow+j]);
		}

		// ensure matrix is diagonally dominant, ie. sum of magnitudes of other entries is
		// less than or equal to the magnitude of the diagonal entry.
		if(diagdom && i < n)
			A[irow+i] = sigma + drand48();

		irow += stride;
		b[i] = drand48();
	}
}

void
init_guess(double *x0, int m)
{
	int i;
	for(i = 0; i < m; i++)
		x0[i] = 0.0;
}

void
iterate_sor(double *A, int m, int n, int stride, double *b, double *x0, double *x1, double w)
{
	double maxres;
	int i;

	for(i = 0; i < 1000; i++){
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
		if(maxres < 1e-6)
			break;
	}
//	printf("%d: maxres %.20f\n", i, maxres);
}

void
square_test(int input_n)
{
	double *A, *b, *x0, *x1;
	double maxres;
	int i, m, n, stride;
	int loop;

	m = input_n;
	n = input_n;
	fprintf(stderr, "square_test %dx%d\n", m, n);

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(n * sizeof b[0]);
	x0 = malloc(n * sizeof x0[0]);
	x1 = malloc(n * sizeof x1[0]);

	for(loop = 0; loop < nloops; loop++){
		build_system(A, m, n, stride, b, 1);
		init_guess(x0, n);
		iterate_sor(A, m, n, stride, b, x0, x0, 1.0);

		relax_ab(A, m, n, stride, x0, x1);
		maxres = fabs(x1[0]-b[0]);
		for(i = 0; i < m; i++)
			maxres = fabs(maxres) > fabs(x1[i]-b[i]) ? fabs(maxres) : fabs(x1[i]-b[i]);
		printf("maxres %.20f\n", maxres);
	}

	free(A);
	free(b);
	free(x0);
	free(x1);
}

void
lsq_test(int input_n)
{
	double *A, *b, *x0;
	double *C, *c;
	double maxres;
	int i, m, n, stride;
	int loop;

	m = input_n + lrand48()%input_n;
	n = input_n;
	fprintf(stderr, "lsq_test %dx%d\n", m, n);

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(m * sizeof b[0]);
	x0 = malloc(m * sizeof x0[0]);

	C = malloc(n * n * sizeof C[0]);
	c = malloc(m * sizeof c[0]);

	for(loop = 0; loop < nloops; loop++){
		build_system(A, m, n, stride, b, 0);
		relax_ata(A, m, n, stride, C, n); // this is the slow step.
		relax_atb(A, m, n, stride, b, c);
		init_guess(x0, n);
		iterate_sor(C, n, n, n, c, x0, x0, 1.0);

		relax_ab(A, m, n, stride, x0, c);
		maxres = fabs(c[0]-b[0]);
		for(i = 0; i < m; i++)
			maxres = fabs(maxres) > fabs(c[i]-b[i]) ? fabs(maxres) : fabs(c[i]-b[i]);
		printf("maxres %.20f\n", maxres);
	}

	free(A);
	free(b);
	free(x0);
	free(C);
	free(c);
}

void
lsq_test2(int input_n)
{
	double *A, *b, *x0;
	double *C, *c;
	double maxres;
	int i, m, n, stride;
	int loop;

	m = input_n + lrand48()%input_n;
	n = input_n;
	fprintf(stderr, "lsq_test2 %dx%d\n", m, n);

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(m * sizeof b[0]);
	x0 = malloc(m * sizeof x0[0]);

	C = malloc(n * n * sizeof C[0]);
	c = malloc(m * sizeof c[0]);

	for(loop = 0; loop < nloops; loop++){
		build_system(A, m, n, stride, b, 0);
		relax_ata(A, m, n, stride, C, n); // this is the slow step.
		relax_atb(A, m, n, stride, b, c);


		for(i = 0; i < n; i++)
			x0[i] = c[i];
		feclearexcept(FE_ALL_EXCEPT);
		relax_solve(C, n, n, n, x0);
		if(fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT)){
			fprintf(stderr, "iteration %d:\n", loop);
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

		relax_ab(A, m, n, stride, x0, c);
		maxres = fabs(c[0]-b[0]);
		for(i = 0; i < m; i++)
			maxres = fabs(maxres) > fabs(c[i]-b[i]) ? fabs(maxres) : fabs(c[i]-b[i]);
		printf("maxres %.20f\n", maxres);
	}

	free(A);
	free(b);
	free(x0);
	free(C);
	free(c);
}


void
minnorm_test(int input_n)
{
	double *A, *b, *x0, *x1;
	double *C, *c;
	double maxres;
	int i, m, n, stride;
	int loop;

	m = input_n;
	n = input_n + lrand48()%input_n;
	fprintf(stderr, "minnorm_test %dx%d\n", m, n);

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(n * sizeof b[0]);
	x0 = malloc(n * sizeof x0[0]);
	if(jacobi)
		x1 = malloc(n * sizeof x1[0]);
	else
		x1 = x0;

	C = malloc(m * m * sizeof C[0]);
	c = malloc(n * sizeof c[0]);

	for(loop = 0; loop < nloops; loop++){
		build_system(A, m, n, stride, b, 0);
		relax_aat(A, m, n, stride, C, m); // this is the slow step.
		init_guess(x0, m);
		iterate_sor(C, m, m, m, b, x0, x1, jacobi ? 0.6 : 1.0);
		relax_atb(A, m, n, stride, x1, c);

		relax_ab(A, m, n, stride, c, x0);
		maxres = fabs(x0[0]-b[0]);
		for(i = 0; i < m; i++)
			maxres = fabs(maxres) > fabs(x0[i]-b[i]) ? fabs(maxres) : fabs(x0[i]-b[i]);
		printf("maxres %.20f\n", maxres);
	}

	free(A);
	free(b);
	free(x0);
	if(jacobi)
		free(x1);
	free(C);
	free(c);
}

void
minnorm_test2(int input_n)
{
	double *A, *b, *x0, *x1;
	double *C, *c;
	double maxres;
	int i, m, n, stride;
	int loop;

	m = input_n;
	n = input_n + lrand48()%input_n;
	fprintf(stderr, "minnorm_test2 %dx%d\n", m, n);

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(n * sizeof b[0]);
	x0 = malloc(n * sizeof x0[0]);
	if(jacobi)
		x1 = malloc(n * sizeof x1[0]);
	else
		x1 = x0;

	C = malloc(m * m * sizeof C[0]);
	c = malloc(n * sizeof c[0]);

	for(loop = 0; loop < nloops; loop++){
		build_system(A, m, n, stride, b, 0);
		relax_aat(A, m, n, stride, C, m); // this is the slow step.

		for(i = 0; i < m; i++)
			x0[i] = b[i];
		feclearexcept(FE_ALL_EXCEPT);
		relax_solve(C, m, m, m, x0);
		if(fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT)){
			fprintf(stderr, "iteration %d:\n", loop);
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
		relax_atb(A, m, n, stride, x0, c);

		relax_ab(A, m, n, stride, c, x0);
		maxres = fabs(x0[0]-b[0]);
		for(i = 0; i < m; i++)
			maxres = fabs(maxres) > fabs(x0[i]-b[i]) ? fabs(maxres) : fabs(x0[i]-b[i]);
		printf("maxres %.20f\n", maxres);
	}

	free(A);
	free(b);
	free(x0);
	if(jacobi)
		free(x1);
	free(C);
	free(c);
}



int
main(int argc, char *argv[])
{
	struct timeval tval;

	if(argc > 1 && !strcmp(argv[1], "jacobi"))
		jacobi++;

	gettimeofday(&tval, NULL);
	srand48(tval.tv_sec ^ tval.tv_usec);

	square_test(500);
	lsq_test(300);
	lsq_test2(300);
	minnorm_test(300);
	minnorm_test2(300);

	return 0;
}