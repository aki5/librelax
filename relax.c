#include <math.h>


double
relax_step(double *A, int m, int n, int stride, double *b, double *x0, double *x1, double *res, double w)
{
	double sigma, rowres, maxres;
	int i, j, irow;

	maxres = 0.0;
	irow = 0;
	for(i = 0; i < m; i++){

		sigma = 0.0;
		rowres = 0.0;
		for(j = 0; j < n; j++){
			double tmp;
			tmp = A[irow+j] * x0[j];
			rowres += tmp;
			if(j != i)
				sigma += tmp;
		}

		x1[i] = (1.0 - w)*x0[i] + w*(b[i] - sigma) / A[irow+i];

		rowres = b[i] - rowres;
		res[i] = rowres;
		rowres = fabs(rowres);
		maxres = rowres > maxres ? rowres : maxres;

		irow += stride;
	}
	return maxres;
}
