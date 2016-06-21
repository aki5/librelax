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
 *	0.0 < w < 2.0: relaxation factor, w < 1.0 is under-relaxed, w > 1.0 is over-relaxed.
 */
double relax_dense(double *A, int m, int n, int stride, double *b, double *x0, double *x1, double *res, double w);
