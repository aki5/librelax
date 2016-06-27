#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <fenv.h>
#include <float.h>
#include "relax.h"

#define nelem(x) (int)((sizeof(x)/sizeof((x)[0])))

int diagdom = 1;
int nloops = 100;
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
iterate_kacz(double *A, int m, int n, int stride, double *b, double *x0)
{
	double maxres;
	double *res;
	int i, row;

	res = malloc(n * sizeof res[0]);
	for(i = 0; i < m; i++)
		x0[i] = 0.0;

	for(i = 0; i < 10000; i++){
		feclearexcept(FE_ALL_EXCEPT);
		if(relax_kacz(A, m, n, stride, b, x0, lrand48()%m) == -1){
			fprintf(stderr, "relax_kacz failed\n");
			free(res);
			return -1; 
		}
		maxres = relax_maxres(A, m, n, stride, b, x0, res);
		if(fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT)){
			fprintf(stderr,
				"floating point exception:%s%s%s%s%s\n",
				fetestexcept(FE_DIVBYZERO) ? " FE_DIVBYZERO" : "",
				fetestexcept(FE_INEXACT) ? " FE_INEXACT" : "",
				fetestexcept(FE_INVALID) ? " FE_INVALID" : "",
				fetestexcept(FE_OVERFLOW) ? " FE_OVERFLOW" : "",
				fetestexcept(FE_UNDERFLOW) ? " FE_UNDERFLOW" : ""
			);
			free(res);
			return -1;
		}
		if(maxres < 1e-14)
			break;
		//fprintf(stderr, "iterate_kacz maxres %f\n", maxres);
	}
	free(res);
	return 0;
}


int
iterate_gs(double *A, int m, int n, int stride, double *b, double *x0)
{
	double maxres;
	int i;

	for(i = 0; i < m; i++)
		x0[i] = 0.0;

	for(i = 0; i < 10000; i++){
		feclearexcept(FE_ALL_EXCEPT);
		maxres = relax_sor(A, m, n, stride, b, x0, NULL);
		if(fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT)){
			fprintf(stderr,
				"floating point exception:%s%s%s%s%s\n",
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
	return 0;
}

int
kaczmarz(double *A, int m, int n, int stride, double *b, double *x0)
{
	double *C, *c;
	int err;

	err = 0;
	C = NULL;
	c = NULL;
	if(m == n){
		memcpy(x0, b, m * sizeof b[0]);
		if((err = iterate_kacz(A, m, n, stride, b, x0)) == -1){
			err = -1;
			goto err_out;
		}
	} else if(m > n){ // overdetermined, solve for least squares fit
		C = malloc(n * n * sizeof C[0]);
		c = malloc(m * sizeof c[0]);
		relax_ata(A, m, n, stride, C, n);
		relax_atb(A, m, n, stride, b, c);
		if((err = iterate_kacz(C, n, n, n, c, x0)) == -1){
			err = -1;
			goto err_out;
		}
	} else { // underdetermined, solve for minimum norm
		C = malloc(m * m * sizeof C[0]);
		c = malloc(m * sizeof c[0]);
		relax_aat(A, m, n, stride, C, m);
		if((err = iterate_kacz(C, m, m, m, b, c)) == -1){
			err = -1;
			goto err_out;
		}
		relax_atb(A, m, n, stride, c, x0);
	}
err_out:
	if(C != NULL)
		free(C);
	if(c != NULL)
		free(c);
	return err;
}

int
gauss_seidel(double *A, int m, int n, int stride, double *b, double *x0)
{
	double *C, *c;
	int err;

	err = 0;
	C = NULL;
	c = NULL;
	if(m == n){
		memcpy(x0, b, m * sizeof b[0]);
		if((err = iterate_gs(A, m, n, stride, b, x0)) == -1){
			err = -1;
			goto err_out;
		}
	} else if(m > n){ // overdetermined, solve for least squares fit
		C = malloc(n * n * sizeof C[0]);
		c = malloc(m * sizeof c[0]);
		relax_ata(A, m, n, stride, C, n);
		relax_atb(A, m, n, stride, b, c);
		if((err = iterate_gs(C, n, n, n, c, x0)) == -1){
			err = -1;
			goto err_out;
		}
	} else { // underdetermined, solve for minimum norm
		C = malloc(m * m * sizeof C[0]);
		c = malloc(m * sizeof c[0]);
		relax_aat(A, m, n, stride, C, m);
		if((err = iterate_gs(C, m, m, m, b, c)) == -1){
			err = -1;
			goto err_out;
		}
		relax_atb(A, m, n, stride, c, x0);
	}
err_out:
	if(C != NULL)
		free(C);
	if(c != NULL)
		free(c);
	return err;
}


int
direct_gauss(double *A, int m, int n, int stride, double *b, double *x0)
{
	double *C, *c;
	int i, err;

	err = 0;
	C = NULL;
	c = NULL;
	if(m == n){
		C = malloc(m * stride * sizeof C[0]);
		memcpy(C, A, m * stride * sizeof A[0]);
		memcpy(x0, b, m * sizeof b[0]);
		if((err = relax_gauss(C, m, n, stride, x0)) == -1){
			fprintf(stderr, "relax_gauss: could not solve\n");
			err = -1;
			goto err_out;
		}
	} else if(m > n){ // overdetermined, solve for least squares fit

		C = malloc(n * n * sizeof C[0]);
		relax_ata(A, m, n, stride, C, n);
		relax_atb(A, m, n, stride, b, x0);
		if((err = relax_gauss(C, n, n, n, x0)) == -1){
			fprintf(stderr, "relax_gauss: could not solve\n");
			err = -1;
			goto err_out;
		}
	} else { // underdetermined, solve for minimum norm
		C = malloc(m * m * sizeof C[0]);
		c = malloc(m * sizeof c[0]);
		relax_aat(A, m, n, stride, C, m);
		for(i = 0; i < m; i++)
			c[i] = b[i];
		if((err = relax_gauss(C, m, m, m, c)) == -1){
			fprintf(stderr, "relax_gauss: could not solve\n");
			err = -1;
			goto err_out;
		}
		relax_atb(A, m, n, stride, c, x0);
	}
err_out:
	if(C != NULL)
		free(C);
	if(c != NULL)
		free(c);
	return err;
}

int
direct_svd(double *A, int m, int n, int stride, double *b, double *x0)
{
	double *U, *V, *w, *tmpvec;
	int ustride, vstride;
	int err;

	if(m >= n){
		int i, j;

		ustride = n;
		vstride = n;
		U = malloc(m * ustride * sizeof U[0]);
		V = malloc(n * vstride * sizeof V[0]);
		w = malloc(n * sizeof w[0]);
		tmpvec = malloc(m * sizeof tmpvec[0]);

		// copy A into U.
		for(i = 0; i < m; i++){
			for(j = 0; j < n; j++)
				U[i*ustride+j] = A[i*stride+j];
			for(; j < ustride; j++)
				U[i*ustride+j] = 0.0;
		}
		err = relax_svd(U, m, n, ustride, V, vstride, w, tmpvec);
		relax_pinvb(U, m, n, ustride, V, vstride, w, b, x0, tmpvec);

		free(U);
		free(V);
		free(w);
		free(tmpvec);
	} else {
		int i, j;

		// swap so that m is the bigger one.
		i = m;
		m = n;
		n = i;

		ustride = n;
		vstride = n;
		U = malloc(m * ustride * sizeof U[0]);
		V = malloc(n * vstride * sizeof V[0]);
		w = malloc(n * sizeof w[0]);
		tmpvec = malloc(m * sizeof tmpvec[0]);

		// transpose A into U.
		for(i = 0; i < m; i++){
			for(j = 0; j < n; j++)
				U[i*ustride+j] = A[j*stride+i];
			for(; j < ustride; j++)
				U[i*ustride+j] = 0.0;
		}
		err = relax_svd(U, m, n, ustride, V, vstride, w, tmpvec);
		relax_pinvtb(U, m, n, ustride, V, vstride, w, b, x0, tmpvec);

		free(U);
		free(V);
		free(w);
		free(tmpvec);
	}

	return err;
}

int (*solvers[])(double *A, int m, int n, int stride, double *b, double *x0) = {
	kaczmarz,
	gauss_seidel,
	direct_gauss,
	direct_svd,
};

char *solver_names[] = {
	"kaczmarz",
	"gauss_seidel",
	"direct_gauss",
	"direct_svd",
};


int
random_test(int input_n, int input_m)
{
	double *A, *b, *x0, *res;
	double maxres;
	int si, m, n, stride;
	int err;

	err = 0;
	m = input_m;
	n = input_n;

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(m * sizeof b[0]);
	x0 = malloc(n * sizeof x0[0]);
	res = malloc(m * sizeof res[0]);

	// A[m][n] * x[n] = b[m]
	build_system(A, m, n, stride, b);
	for(si = 0; si < nelem(solvers); si++){
		if((err = (*solvers[si])(A, m, n, stride, b, x0)) == -1)
			continue;
		maxres = relax_maxres(A, m, n, stride, b, x0, res);
		printf("%dx%d %-12s maxres %.20f\n", m, n, solver_names[si], maxres);
	}

	free(A);
	free(b);
	free(x0);
	free(res);
	return err;
}

int
main(void)
{
	struct timeval tval;
	int loop, rndmax;

	gettimeofday(&tval, NULL);
	srand48(tval.tv_sec ^ tval.tv_usec);

	rndmax = 50;
	for(loop = 0; loop < nloops; loop++){
		random_test(rndmax + lrand48()%rndmax, rndmax + lrand48()%rndmax);
		printf("\n");
	}

	return 0;
}
