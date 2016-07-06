#include <stddef.h>
#include <math.h>
#include "relax.h"

/*
 *	The Conjugate Gradient iteration solves Ax = b for x
 */
double
relax_conjgrad(double *A, int m, int n, int stride, double *x0, double *res, double *dir, double *adir)
{
	double restmp, maxres;
	double alpha, beta, delta;
	int i;

	relax_ab(A, m, n, stride, dir, adir);

	delta = relax_dot(res, 1, res, 1, n);
	alpha = delta / relax_dot(dir, 1, adir, 1, n);

	x0[0] = x0[0] + alpha*dir[0];
	res[0] = res[0] - alpha*adir[0];
	maxres = fabs(res[0]);
	for(i = 1; i < m; i++){
		x0[i] = x0[i] + alpha*dir[i];
		res[i] = res[i] - alpha*adir[i];
		restmp = fabs(res[i]);
		maxres = maxres > restmp ? maxres : restmp;
	}

	// this dot product will be recomputed at the start of next iteration..
	// should introduce another argument for it (delta).
	beta = relax_dot(res, 1, res, 1, n) / delta;
	for(i = 0; i < m; i++){
		dir[i] = res[i] + beta*dir[i];
	}

	return maxres;
}
