#include <math.h>

/*
 *	over-relaxation step for a dense M-by-N matrix
 *
 *	there are M rows and N columns in the M-by-N matrix A.
 *	i indicates the row (0 to m-1), and
 *	j indicates the column (0 to n-1)
 *
 *	x0 == x1: gauss-seidel method (overwrite in-place)
 *	x0 != x1: jacobi method (separate output vector)
 *	w == 1.0: classic jacobi or gauss-seidel.
 *	0.0 < w < 2.0: over-relaxation,
 */
double
relax_step(double *A, int m, int n, int stride, double *b, double *x0, double *x1, double *res, double w)
{
	double sigma, resval, maxres;
	int i, j, irow;

	maxres = 0.0;
	irow = 0;
	for(i = 0; i < m; i++){

		sigma = 0.0;
		resval = 0.0;
		for(j = 0; j < n; j++){
			double tmp;
			tmp = A[irow+j] * x0[j];
			resval += tmp;
			if(j != i)
				sigma += tmp;
		}

		x1[i] = (1.0 - w)*x0[i] + w*(b[i] - sigma) / A[irow+i];

		resval = b[i] - resval;
		res[i] = resval;
		resval = fabs(resval);
		maxres = resval > maxres ? resval : maxres;

		irow += stride;
	}
	return maxres;
}
