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
double relax_sor(double *A, int m, int n, int stride, double *b, double *x0, double *x1, double *res, double w);

/*
 *	relax_solve destructively soves the system Ax = b for x.
 *	After returning, A contains a diagonal matrix and b contains
 * 	the vector x.
 *
 *	If the matrix is found to be singular, relax_solve returns -1.
 */
int relax_solve(double *A, int m, int n, int stride, double *b);

/*
 *	When it turns out that we have too many equations (M > N), it is
 *	possible to reduce the number of rows to N by multiplying both,
 *	the matrix A and vector b by the transpose of A.
 *
 *	Solving the resulting NxN system for x results in a least squares
 *	fit to the original overdetermined system.
 *
 *	This outrageous claim follows fairly simply from taking a gradient
 *	of the squared length of residual of the system, and then noticing
 *	that a positive square function has its minimum value when all
 *	derivatives are zero.
 *
 *	The solution could also be expressed in matrix form, x = A+ * b,
 *	where A+ == (At*A)^-1 * At is also known as Moore-Penrose pseudoinverse.
 *
 *	relax_ata multiplies A by its own transpose and store the result in C
 *	A is MxN (has m rows and n columns)
 *	C is NxN (has n rows and n columns)
 *
 *	relax_atb multiplies b by transpose of A, store the result in c
 *	A is MxN (has m rows and n columns)
 *	c is Nx1 (has n rows)
 */
void relax_ata(double *A, int m, int n, int astride, double *C, int cstride);
void relax_aat(double *A, int m, int n, int astride, double *C, int cstride);
void relax_atb(double *A, int m, int n, int astride, double *b, double *c);
void relax_ab(double *A, int m, int n, int astride, double *b, double *c);
