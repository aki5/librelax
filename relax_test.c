#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <fenv.h>
#include <float.h>
#include "relax.h"

#define nelem(x) (int)((sizeof(x)/sizeof((x)[0])))
#define TOLERANCE 1e-12

int diagdom = 0;
int nloops = 100;
int maxiter = 100000;
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
			//A[irow+j] = 2.0*drand48()-1.0;
			A[irow+j] = drand48();
			if(i != j)
				sigma += fabs(A[irow+j]);
		}

		// ensure matrix is diagonally dominant, ie. sum of magnitudes of other entries is
		// less than or equal to the magnitude of the diagonal entry.
		if(diagdom && i < n)
			A[irow+i] = sigma + drand48();

		irow += stride;
		//b[i] = 2.0*drand48()-1.0;
		b[i] = drand48();
	}
}


int
iterate_kacz(double *A, int m, int n, int stride, double *b, double *x0)
{
	double maxres;
	double *res;
	int i;
	//int j;

	res = malloc(n * sizeof res[0]);
	for(i = 0; i < m; i++)
		x0[i] = 0.0;

	for(i = 0; i < maxiter; i++){
		feclearexcept(FE_ALL_EXCEPT);
		//for(j = 0; j < m; j++){
			if(relax_kacz(A, m, n, stride, b, x0, lrand48()%m, 1.0) == -1){
			//if(relax_kacz(A, m, n, stride, b, x0, j) == -1){
				fprintf(stderr, "relax_kacz failed\n");
				free(res);
				return -1;
			}
		//}
		if(fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT)){
			printf(
				"%s%s%s%s%s\n",
				fetestexcept(FE_DIVBYZERO) ? " FE_DIVBYZERO" : "",
				fetestexcept(FE_INEXACT) ? " FE_INEXACT" : "",
				fetestexcept(FE_INVALID) ? " FE_INVALID" : "",
				fetestexcept(FE_OVERFLOW) ? " FE_OVERFLOW" : "",
				fetestexcept(FE_UNDERFLOW) ? " FE_UNDERFLOW" : ""
			);
			free(res);
			return -1;
		}

		maxres = relax_maxres(A, m, n, stride, x0, b, res);
		if(maxres < TOLERANCE)
			break;
	}
	free(res);
	printf(" iter %7d", i);

	return 0;
}


int
iterate_gs(double *A, int m, int n, int stride, double *b, double *x0)
{
	double maxres;
	int i;

	for(i = 0; i < m; i++)
		x0[i] = 0.0;

	for(i = 0; i < maxiter; i++){
		feclearexcept(FE_ALL_EXCEPT);
		maxres = relax_coordesc(A, m, n, stride, b, x0, NULL);
		if(fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT)){
			printf(
				"%s%s%s%s%s\n",
				fetestexcept(FE_DIVBYZERO) ? " FE_DIVBYZERO" : "",
				fetestexcept(FE_INEXACT) ? " FE_INEXACT" : "",
				fetestexcept(FE_INVALID) ? " FE_INVALID" : "",
				fetestexcept(FE_OVERFLOW) ? " FE_OVERFLOW" : "",
				fetestexcept(FE_UNDERFLOW) ? " FE_UNDERFLOW" : ""
			);
			return -1;
		}
		if(maxres < TOLERANCE)
			break;
	}
	printf(" iter %7d", i);
	return 0;
}

int
iterate_graddesc(double *A, int m, int n, int stride, double *b, double *x0, double *res, double *ares)
{
	double maxres;
	int i;

	for(i = 0; i < m; i++)
		x0[i] = 0.0;

	maxres = relax_maxres(A, m, n, stride, x0, b, res);
	for(i = 0; i < maxiter; i++){
		feclearexcept(FE_ALL_EXCEPT);
		maxres = relax_graddesc(A, m, n, stride, x0, res, ares);
		if(fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT)){
			printf(
				"%s%s%s%s%s\n",
				fetestexcept(FE_DIVBYZERO) ? " FE_DIVBYZERO" : "",
				fetestexcept(FE_INEXACT) ? " FE_INEXACT" : "",
				fetestexcept(FE_INVALID) ? " FE_INVALID" : "",
				fetestexcept(FE_OVERFLOW) ? " FE_OVERFLOW" : "",
				fetestexcept(FE_UNDERFLOW) ? " FE_UNDERFLOW" : ""
			);
			return -1;
		}
		if(maxres < TOLERANCE)
			if(relax_maxres(A, m, n, stride, x0, b, res) < TOLERANCE)
				break;
	}
	printf(" iter %7d", i);
	return 0;
}


int
iterate_conjgrad(double *A, int m, int n, int stride, double *b, double *x0, double *res, double *dir, double *adir, double *tdir)
{
	double reslen2, maxres;
	int i;

	for(i = 0; i < n; i++)
		x0[i] = 0.0;

	if(m > n){
		relax_cgls_init(A, m, n, stride, x0, b, res, dir, tdir, &reslen2);
		for(i = 0; i < maxiter; i++){
			feclearexcept(FE_ALL_EXCEPT);
			maxres = relax_cgls(A, m, n, stride, x0, res, dir, adir, tdir, &reslen2);
			if(fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT)){
				printf(
					"%s%s%s%s%s\n",
					fetestexcept(FE_DIVBYZERO) ? " FE_DIVBYZERO" : "",
					fetestexcept(FE_INEXACT) ? " FE_INEXACT" : "",
					fetestexcept(FE_INVALID) ? " FE_INVALID" : "",
					fetestexcept(FE_OVERFLOW) ? " FE_OVERFLOW" : "",
					fetestexcept(FE_UNDERFLOW) ? " FE_UNDERFLOW" : ""
				);
				return -1;
			}
			if(maxres < TOLERANCE)
				if(relax_maxres(A, m, n, stride, x0, b, res) < TOLERANCE)
					break;
		}
	} else {
		relax_conjgrad_init(A, m, n, stride, x0, b, res, dir, &reslen2);
		for(i = 0; i < maxiter; i++){
			feclearexcept(FE_ALL_EXCEPT);
			maxres = relax_conjgrad(A, m, n, stride, x0, res, dir, adir, &reslen2);
			if(fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT)){
				printf(
					"%s%s%s%s%s\n",
					fetestexcept(FE_DIVBYZERO) ? " FE_DIVBYZERO" : "",
					fetestexcept(FE_INEXACT) ? " FE_INEXACT" : "",
					fetestexcept(FE_INVALID) ? " FE_INVALID" : "",
					fetestexcept(FE_OVERFLOW) ? " FE_OVERFLOW" : "",
					fetestexcept(FE_UNDERFLOW) ? " FE_UNDERFLOW" : ""
				);
				return -1;
			}
			if(maxres < TOLERANCE)
				if(relax_maxres(A, m, n, stride, x0, b, res) < TOLERANCE)
					break;
		}
	}
	printf(" iter %7d", i);
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
coordinate_descent(double *A, int m, int n, int stride, double *b, double *x0)
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
gradient_descent(double *A, int m, int n, int stride, double *b, double *x0)
{
	double *C, *c, *res, *ares;
	int err;

	err = 0;
	C = NULL;
	c = NULL;
	if(m == n){
		memcpy(x0, b, m * sizeof b[0]);
		ares = malloc(m * sizeof ares[0]);
		res = malloc(m * sizeof res[0]);
		if((err = iterate_graddesc(A, m, n, stride, b, x0, res, ares)) == -1){
			err = -1;
			goto err_out;
		}
	} else if(m > n){ // overdetermined, solve for least squares fit
		C = malloc(n * n * sizeof C[0]);
		c = malloc(m * sizeof c[0]);
		ares = malloc(n * sizeof ares[0]);
		res = malloc(n * sizeof res[0]);
		relax_ata(A, m, n, stride, C, n);
		relax_atb(A, m, n, stride, b, c);
		if((err = iterate_graddesc(C, n, n, n, c, x0, res, ares)) == -1){
			err = -1;
			goto err_out;
		}
	} else { // underdetermined, solve for minimum norm
		C = malloc(m * m * sizeof C[0]);
		c = malloc(m * sizeof c[0]);
		ares = malloc(m * sizeof ares[0]);
		res = malloc(m * sizeof res[0]);
		relax_aat(A, m, n, stride, C, m);
		if((err = iterate_graddesc(C, m, m, m, b, c, res, ares)) == -1){
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
	free(res);
	free(ares);
	return err;
}

int
conjugate_gradient(double *A, int m, int n, int stride, double *b, double *x0)
{
	double *C, *c, *res, *dir, *adir, *tdir;
	int err;

	err = 0;
	C = NULL;
	c = NULL;
	tdir = NULL;
	if(m == n){
		adir = malloc(m * sizeof adir[0]);
		dir = malloc(m * sizeof dir[0]);
		res = malloc(m * sizeof res[0]);
		if((err = iterate_conjgrad(A, m, n, stride, b, x0, res, dir, adir, tdir)) == -1){
			err = -1;
			goto err_out;
		}
	} else if(m > n){ // overdetermined, solve for least squares fit
		tdir = malloc(m * sizeof tdir[0]);
		adir = malloc(m * sizeof adir[0]);
		dir = malloc(m * sizeof dir[0]);
		res = malloc(m * sizeof res[0]);
		if((err = iterate_conjgrad(A, m, n, stride, b, x0, res, dir, adir, tdir)) == -1){
			err = -1;
			goto err_out;
		}
	} else { // underdetermined, solve for minimum norm
		C = malloc(m * m * sizeof C[0]);
		c = malloc(m * sizeof c[0]);
		adir = malloc(m * sizeof adir[0]);
		dir = malloc(m * sizeof dir[0]);
		res = malloc(m * sizeof res[0]);
		relax_aat(A, m, n, stride, C, m);
		if((err = iterate_conjgrad(C, m, m, m, b, c, res, dir, adir, tdir)) == -1){
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
	if(tdir != NULL)
		free(tdir);
	free(res);
	free(dir);
	free(adir);

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

	printf(" iter %7d", -1);
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

	printf(" iter %7d", -1);
	return err;
}

int (*solvers[])(double *A, int m, int n, int stride, double *b, double *x0) = {
	kaczmarz,
	gradient_descent,
	coordinate_descent,
	conjugate_gradient,
	direct_gauss,
	direct_svd,
};

char *solver_names[] = {
	"kaczmarz",
	"gradient_descent",
	"coordinate_descent",
	"conjugate_gradient",
	"direct_gauss",
	"direct_svd",
};

int64_t
nsec(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return ((int64_t)tv.tv_sec * 1000000000) + (int64_t)tv.tv_usec*1000;
}

int
random_test(int input_n, int input_m)
{
	double *A, *b, *x0, *res;
	double maxres;
	int64_t stime, etime;
	int si, m, n, stride;
	int err;

	err = 0;

#if 0
	if(input_m < input_n){
		m = input_m;
		n = input_n;
	} else {
		m = input_n;
		n = input_m;
	}
#else
	m = input_n;
	n = input_m;
#endif

	stride = n;
	A = malloc(m * stride * sizeof A[0]);
	b = malloc(m * sizeof b[0]);
	x0 = malloc(n * sizeof x0[0]);
	res = malloc(m * sizeof res[0]);

	// A[m][n] * x[n] = b[m]
	build_system(A, m, n, stride, b);
	for(si = 0; si < nelem(solvers); si++){
		printf("%dx%d %-18s", m, n, solver_names[si]);
		fflush(stdout);
		stime = nsec();
		if((err = (*solvers[si])(A, m, n, stride, b, x0)) == -1)
			continue;
		etime = nsec();
		maxres = relax_maxres(A, m, n, stride, x0, b, res);
		printf(" maxres %.16f time %.6f\n", maxres, 1e-9*(etime-stime));
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
