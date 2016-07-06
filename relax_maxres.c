
#include <math.h>
#include "relax.h"

double
relax_maxres(double *A, int m, int n, int stride, double *b, double *x, double *tmp)
{
	double res, maxres;
	int i;

	relax_ab(A, m, n, stride, x, tmp);

	tmp[0] = b[0] - tmp[0];
	maxres = fabs(tmp[0]);
	for(i = 1; i < m; i++){
		tmp[i] = b[i] - tmp[i];
		res = fabs(tmp[i]);
		maxres = maxres > res ? maxres : res;
	}

	return maxres;
}
