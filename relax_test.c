#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <fenv.h>
#include <float.h>
#include "relax.h"

#define nelem(x) (int)((sizeof(x)/sizeof((x)[0])))

int diagdom = 0;
int nloops = 10;
int (*solvers[])(double *A, int m, int n, int stride, double *b, double *x0);

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
build_system(double *A, int m, int n, int stride, double *b)
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

int
iterative_gs(double *A, int m, int n, int stride, double *b, double *x0)
{
	double maxres;
	int i;

	for(i = 0; i < m; i++)
		x0[i] = 0.0;

	for(i = 0; i < 1000; i++){
		feclearexcept(FE_ALL_EXCEPT);
		maxres = relax_sor(A, m, n, stride, b, x0, x0, NULL, 1.0); // for gauss-seidel
		if(fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT)){
			fprintf(stderr,
				"iterative_gs: floating point exception:%s%s%s%s%s\n",
				fetestexcept(FE_DIVBYZERO) ? " FE_DIVBYZERO" : "",
				fetestexcept(FE_INEXACT) ? " FE_INEXACT" : "",
				fetestexcept(FE_INVALID) ? " FE_INVALID" : "",
				fetestexcept(FE_OVERFLOW) ? " FE_OVERFLOW" : "",
				fetestexcept(FE_UNDERFLOW) ? " FE_UNDERFLOW" : ""
			);
			return -1;
		}
		if(maxres < 1e-14)
			break;
	}
//	printf("%d: maxres %.20f\n", i, maxres);
	return 0;
}

int
direct_gauss(double *A, int m, int n, int stride, double *b, double *x0)
{
	double *C;
	int err;

	err = 0;
	memcpy(x0, b, m * sizeof b[0]);
	C = malloc(m * stride * sizeof C[0]);
	memcpy(C, A, m * stride * sizeof A[0]);
	feclearexcept(FE_ALL_EXCEPT);
	if((err = relax_gauss(C, m, n, stride, x0)) == -1){
		fprintf(stderr, "relax_gauss: matrix is singular\n");
		goto err_out;
	}
err_out:
	free(C);
	return err;
}

int (*solvers[])(double *A, int m, int n, int stride, double *b, double *x0) = {
	iterative_gs,
	direct_gauss,
};

char *solver_names[] = {
	"iterative_gs",
	"direct_gauss",
};

int
square_test(int input_n)
{
	double *A, *b, *x0, *x1;
	double maxres;
	int si, m, n, stride;
	int loop;
	int err;

	err = 0;
	m = input_n;
	n = input_n;
	fprintf(stderr, "square_test %dx%d\n", m, n);

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(n * sizeof b[0]);
	x0 = malloc(n * sizeof x0[0]);
	x1 = malloc(n * sizeof x1[0]);

	for(loop = 0; loop < nloops; loop++){
		build_system(A, m, n, stride, b);
		for(si = 0; si < nelem(solvers); si++){
			if((err = (*solvers[si])(A, m, n, stride, b, x0)) == -1)
				continue;
			maxres = relax_maxres(A, m, n, stride, b, x0, x1);
			printf("%s: maxres %.20f\n", solver_names[si], maxres);
		}
	}

	free(A);
	free(b);
	free(x0);
	free(x1);
	return err;
}

int
lsq_test(int input_n)
{
	double *A, *b, *x0, *x1;
	double *C, *c;
	double maxres;
	int si, m, n, stride;
	int loop;
	int err;

	err = 0;

	m = input_n + 1 + lrand48()%input_n;
	n = input_n;
	fprintf(stderr, "lsq_test %dx%d\n", m, n);

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(m * sizeof b[0]);
	x0 = malloc(m * sizeof x0[0]);
	x1 = malloc(m * sizeof x1[0]);

	C = malloc(n * n * sizeof C[0]);
	c = malloc(m * sizeof c[0]);

	for(loop = 0; loop < nloops; loop++){
		build_system(A, m, n, stride, b);
		for(si = 0; si < nelem(solvers); si++){
			relax_ata(A, m, n, stride, C, n); // this is the slow step.
			relax_atb(A, m, n, stride, b, c);
			if((err = (*solvers[si])(C, n, n, n, c, x0)) == -1)
				continue;
			maxres = relax_maxres(A, m, n, stride, b, x0, c);
			printf("%s: maxres %.20f\n", solver_names[si], maxres);
		}
	}

	free(A);
	free(b);
	free(x0);
	free(x1);
	free(C);
	free(c);
	return err;
}

int
minnorm_test(int input_n)
{
	double *A, *b, *x0;
	double *C, *c;
	double maxres;
	int si, m, n, stride;
	int loop;
	int err;

	err = 0;
	m = input_n;
	n = input_n + 1 + lrand48()%input_n;
	fprintf(stderr, "minnorm_test %dx%d\n", m, n);

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(n * sizeof b[0]);
	x0 = malloc(n * sizeof x0[0]);

	C = malloc(m * m * sizeof C[0]);
	c = malloc(n * sizeof c[0]);

	for(loop = 0; loop < nloops; loop++){
		build_system(A, m, n, stride, b);
		for(si = 0; si < nelem(solvers); si++){
			relax_aat(A, m, n, stride, C, m); // this is the slow step.
			if((err = (*solvers[si])(C, m, m, m, b, x0)) == -1)
				continue;
			relax_atb(A, m, n, stride, x0, c);

			maxres = relax_maxres(A, m, n, stride, b, c, x0);
			printf("%s: maxres %.20f\n", solver_names[si], maxres);
		}
	}

	free(A);
	free(b);
	free(x0);
	free(C);
	free(c);
	return err;
}

int
svd_test(int m, int n, int debug)
{
	double *rnd;
	double *u;
	double *v;
	double *w;
	double *tmpvec;
	int i, j, stride;
	int loop;
	int err;

	fprintf(stderr, "svd_test direct %dx%d\n", m, n);

	err = 0;
	stride = n;
	rnd = malloc(m * stride * sizeof rnd[0]);
	u = malloc(m * stride * sizeof u[0]);
	v = malloc(n * stride * sizeof v[0]);
	w = malloc(n * sizeof w[0]);
	tmpvec = malloc(n * sizeof tmpvec[0]);

	for(loop = 0; loop < nloops; loop++){
		for(i = 0; i < m*stride; i++){
			rnd[i] = drand48()*drand48()*drand48()*drand48();
			if(drand48() < 0.5)
				rnd[i] = 0.0;
			if(drand48() < 0.5)
				rnd[i] = -rnd[i];
		}

		memcpy(u, rnd, m * stride * sizeof rnd[0]);
		err = relax_svd(u, m, n, n, v, n, w, tmpvec);
		if(err == -1){
			fprintf(stderr, "relax_svd did not converge\n");
			goto err_out;
		}

		if(debug != 0){
			for(j = 0; j < m; j++){
				double *row = u + j*stride;
				printf("u%02d:", j);
				for(i = 0; i < n; i++)
					printf(" %+6.3f", row[i]);
				printf("\n");
			}

			printf("\n");

			for(j = 0; j < n; j++){
				double *row = v + j*stride;
				printf("v%02d:", j);
				for(i = 0; i < n; i++)
					printf(" %+6.3f", row[i]);
				printf("\n");
			}

			printf("\n");

			printf("w00:");
			for(i = 0; i < n; i++){
				printf(" %+9.6f", w[i]);
			}
			printf("\n");
		}

		printf("svd loop %d\n", loop);
	}

err_out:
	free(rnd);
	free(u);
	free(v);
	free(w);
	free(tmpvec);
	return err;
}

int
main(void)
{
	struct timeval tval;

	gettimeofday(&tval, NULL);
	srand48(tval.tv_sec ^ tval.tv_usec);

	square_test(300);
	printf("\n");
	lsq_test(300);
	printf("\n");
	minnorm_test(300);
	printf("\n");

	svd_test(300 + lrand48()%300, 300, 0);
//	svd_test(20, 20, 1);


	return 0;
}
