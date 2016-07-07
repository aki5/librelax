#include <stdio.h>
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
relax_cgls_init(double *A, int m, int n, int stride, double *x, double *b, double *res, double *dir, double *rlen2p)
{
	relax_ab(A, m, n, stride, x, res);
	relax_bypax(-1.0, res, 1, 1.0, b, 1, m);
	relax_atb(A, m, n, stride, res, dir);
	*rlen2p = relax_dot(dir, 1, dir, 1, n);
	return sqrt(*rlen2p);
}

double
relax_cgls(double *A, int m, int n, int stride, double *x0, double *res, double *dir, double *tmp, double *reslen2)
{
	double alpha, beta, gamma;

	relax_ab(A, m, n, stride, dir, tmp); // q = Ap

	gamma = relax_dot(tmp, 1, tmp, 1, m);
	if(gamma <= 0.0)
		fprintf(stderr, "relax_cgls: system not definite\n");
	alpha = *reslen2 / gamma;

	relax_bypax(1.0, x0, 1, alpha, dir, 1, n);
	relax_bypax(1.0, res, 1, -alpha, tmp, 1, m);

	relax_atb(A, m, n, stride, res, tmp);
	gamma = relax_dot(tmp, 1, tmp, 1, n);

	beta =  gamma / *reslen2;
	relax_bypax(beta, dir, 1, 1.0, tmp, 1, n);
	*reslen2 = gamma;

	return sqrt(gamma);
}
