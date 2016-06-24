
#include <math.h>
#include "relax.h"

/*
 *	run gaussian elimination on the MxN matrix A,
 *	destroying it in the process.
 *
 *	at completion, A contains an upper triangular matrix
 *	and b has been replaced with x, the solution to the
 *	original Ax = b
 */
int
relax_gauss(double *A, int m, int n, int stride, double *b)
{
	double tmp, maxpiv;
	int i, j, k, ioff, joff, maxrow;
	int row[m];

	for(i = 0; i < m; i++)
		row[i] = i;

	// put matrix in upper triangular form
	for(i = 0; i < m; i++){
		// find maximum value down the column
		// aka. partial pivoting
		maxpiv = A[row[i]*stride+i];
		maxrow = i;
		for(k = i+1; k < m; k++){
			tmp = A[row[k]*stride+i];
			if(fabs(tmp) > fabs(maxpiv)){
				maxpiv = tmp;
				maxrow = k;
			}
		}

		if(fabs(maxpiv) < 1e-6)
			return -1;

		// swap max row and current row
		ioff = row[i];
		row[i] = row[maxrow];
		row[maxrow] = ioff;
		ioff = row[i]*stride;

		// eliminate head element from rows below
		for(j = i+1; j < m; j++){
			joff = row[j]*stride;
			tmp = -A[joff+i]/maxpiv;
			for(k = i; k < n; k++)
				A[joff+k] += tmp*A[ioff+k];
			b[row[j]] += tmp*b[row[i]];
		}
	}

	// do back-substitution from the bottom up
	// note how similar to the jacobi method this step is
	for(i = m-1; i >= 0; i--){
		tmp = 0.0;
		ioff = row[i]*stride;
		for(j = i+1; j < n; j++)
			tmp += A[ioff+j]*b[row[j]];
		b[row[i]] = (b[row[i]] - tmp) / A[ioff+i];
	}

	return 0;
}
