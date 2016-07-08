#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include "relax.h"

/*
 *	The Conjugate Gradient Minimum Norm solves AAᵀy = b for y.
 *
 *	After the solver has converged, x = Aᵀy is the minimum norm
 *	solution to the underdetermined system Ax = b
 *
 */

double
relax_cgmn_init(double *A, int m, int n, int stride, double *x, double *b, double *res, double *dir, double *rlen2p)
{
	relax_atb(A, m, n, stride, x, dir);
	relax_ab(A, m, n, stride, dir, res);
	relax_bypax(-1.0, res, 1, 1.0, b, 1, m);
	relax_copy(dir, 1, res, 1, m);
	*rlen2p = relax_dot(res, 1, res, 1, m);
	return sqrt(*rlen2p);
}

double
relax_cgmn(double *A, int m, int n, int stride, double *x0, double *res, double *dir, double *tmp, double *tmp2, double *reslen2)
{
	double alpha, beta, gamma;

	relax_atb(A, m, n, stride, dir, tmp2);
	relax_ab(A, m, n, stride, tmp2, tmp);
	gamma = relax_dot(dir, 1, tmp, 1, m);
	alpha = *reslen2 / gamma;

	relax_bypax(1.0, x0, 1, alpha, dir, 1, m);
	relax_bypax(1.0, res, 1, -alpha, tmp, 1, m);

	gamma = relax_dot(res, 1, res, 1, m);
	beta =  gamma / *reslen2;
	relax_bypax(beta, dir, 1, 1.0, res, 1, m);
	*reslen2 = gamma;

	return sqrt(gamma);
}
