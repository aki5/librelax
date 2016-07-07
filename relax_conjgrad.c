#include <stddef.h>
#include <math.h>
#include "relax.h"

/*
 *	The Conjugate Gradient iteration solves Ax = b for x
 */
double
relax_conjgrad_init(double *A, int m, int n, int stride, double *x, double *b, double *tmp, double *cpy, double *rlen2p)
{
	relax_ab(A, m, n, stride, x, tmp);
	relax_bypax(-1.0, tmp, 1, 1.0, b, 1, m);
	relax_copy(cpy, 1, tmp, 1, m);
	*rlen2p = relax_dot(tmp, 1, tmp, 1, m);
	return 0.0;
}


double
relax_conjgrad(double *A, int m, int n, int stride, double *x0, double *res, double *dir, double *adir, double *reslen2)
{
	double restmp, maxres;
	double alpha, beta, gamma;
	int i;

	relax_ab(A, m, n, stride, dir, adir);

	alpha = *reslen2 / relax_dot(dir, 1, adir, 1, n);

	x0[0] = x0[0] + alpha*dir[0];
	res[0] = res[0] - alpha*adir[0];
	maxres = fabs(res[0]);
	for(i = 1; i < m; i++){
		x0[i] = x0[i] + alpha*dir[i];
		res[i] = res[i] - alpha*adir[i];
		restmp = fabs(res[i]);
		maxres = maxres > restmp ? maxres : restmp;
	}

	gamma = relax_dot(res, 1, res, 1, n);
	beta =  gamma / *reslen2;
	*reslen2 = gamma;
	for(i = 0; i < m; i++){
		dir[i] = res[i] + beta*dir[i];
	}

	return maxres;
}
