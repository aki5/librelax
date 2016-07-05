
#include <math.h>
#include "relax.h"

double
relax_maxres(double *A, int m, int n, int stride, double *b, double *x, double *tmp)
{
	double maxres;
	int i;

	relax_ab(A, m, n, stride, x, tmp);
	maxres = fabs(tmp[0]-b[0]);
	for(i = 0; i < m; i++){
		tmp[i] = b[i] - tmp[i];
		maxres = fabs(maxres) > fabs(tmp[i]) ? fabs(maxres) : fabs(tmp[i]);
	}
	return maxres;
}
