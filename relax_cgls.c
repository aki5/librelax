#include <stddef.h>
#include <math.h>
#include "relax.h"

/*
 *	The Conjugate Gradient Least Squares solves AᵀAx = Aᵀb for x
 *
 *	This means that the initial res and dir need to be computed by taking
 *	the transpose into account, for which relax_cgls_init can be used,
 *	passing the in the m-vectors res and dir as atx and atb.
 */

double
relax_cgls_init(double *A, int m, int n, int stride, double *x, double *b, double *atx, double *atb, double *rlen2p)
{
	double res, maxres, reslen2;
	int i;

	relax_ab(A, m, n, stride, x, atb);
	relax_atb(A, m, n, stride, atb, atx);
	relax_atb(A, m, n, stride, b, atb);

	atx[0] = atb[0] - atx[0];
	atb[0] = atx[0];
	reslen2 = atx[0] * atx[0];
	maxres = fabs(atx[0]);
	for(i = 1; i < n; i++){
		atx[i] = atb[i] - atx[i];
		atb[i] = atx[i];
		res = fabs(atx[i]);
		maxres = maxres > res ? maxres : res;
		reslen2 += atx[i] * atx[i];
	}
	*rlen2p = reslen2;

	return maxres;
}

double
relax_cgls(double *A, int m, int n, int stride, double *x0, double *res, double *dir, double *adir, double *tdir, double *reslen2)
{
	double restmp, maxres;
	double alpha, beta, gamma;
	int i;

	relax_ab(A, m, n, stride, dir, tdir);
	relax_atb(A, m, n, stride, tdir, adir);

	alpha = *reslen2 / relax_dot(dir, 1, adir, 1, n);

	x0[0] = x0[0] + alpha*dir[0];
	res[0] = res[0] - alpha*adir[0];
	gamma = res[0]*res[0];
	maxres = fabs(res[0]);
	for(i = 1; i < n; i++){
		x0[i] = x0[i] + alpha*dir[i];
		res[i] = res[i] - alpha*adir[i];
		restmp = fabs(res[i]);
		maxres = maxres > restmp ? maxres : restmp;
		gamma += res[i]*res[i];
	}

	beta =  gamma / *reslen2;
	*reslen2 = gamma;
	for(i = 0; i < n; i++){
		dir[i] = res[i] + beta*dir[i];
	}

	// |Aᵀr| / |A||r| is a measure of fitness
	// that should be useful for least squares.

	return maxres;
}
