#include <stddef.h>
#include <math.h>
#include "relax.h"

/*
 *	The Conjugate Gradient iteration solves Ax = b for x
 */






double
relax_conjgrad_init(double *A, int m, int n, int stride, double *x, double *b, double *res, double *dir, double *rlen2p)
{
	relax_ab(A, m, n, stride, x, res);
	relax_bypax(-1.0, res, 1, 1.0, b, 1, m);
	relax_copy(dir, 1, res, 1, m);
	*rlen2p = relax_dot(res, 1, res, 1, m);
	return sqrt(*rlen2p);
}

double
relax_conjgrad(double *A, int m, int n, int stride, double *x0, double *res, double *dir, double *tmp, double *reslen2)
{
	double alpha, beta, gamma;

	relax_ab(A, m, n, stride, dir, tmp);
	gamma = relax_dot(dir, 1, tmp, 1, n);
	alpha = *reslen2 / gamma;

	relax_bypax(1.0, x0, 1, alpha, dir, 1, m);
	relax_bypax(1.0, res, 1, -alpha, tmp, 1, m);


	gamma = relax_dot(res, 1, res, 1, n);
	beta =  gamma / *reslen2;
	relax_bypax(beta, dir, 1, 1.0, res, 1, n);
	*reslen2 = gamma;

	return sqrt(gamma);
}
