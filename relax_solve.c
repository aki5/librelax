
#include <math.h>
#include "relax.h"

/*
 *	run gauss-jordan elimination on the MxN matrix A,
 *	destroying it in the process.
 */
int
relax_solve(double *A, int m, int n, int stride, double *b)
{
	double tmp, maxpiv;
	int i, j, k, ioff, joff, maxrow;
	int row[m];

	for(i = 0; i < m; i++)
		row[i] = i;

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

		// eliminate head element from all rows
		for(j = 0; j < m; j++){
			if(j == i)
				continue;
			joff = row[j]*stride;
			tmp = -A[joff+i]/maxpiv;
			for(k = i; k < n; k++)
				A[joff+k] += tmp*A[ioff+k];
			b[row[j]] += tmp*b[row[i]];
		}
	}

	// divide A_ii and b_i with A_ii.
	for(i = 0; i < m; i++){
		tmp = A[row[i]*stride+i];
		b[row[i]] /= tmp;
		A[row[i]*stride+i] /= tmp;
	}

	return 0;
}
