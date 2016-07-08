/*
 *	coordinate descent step for a dense M-by-N matrix
 *
 *	there are M rows and N columns in the M-by-N matrix A.
 *	i indicates the row (0 to m-1), and
 *	j indicates the column (0 to n-1)
 *
 */
double relax_coordesc(double *A, int m, int n, int stride, double *b, double *x0, double *res);

/*
 *	kaczmarz iteration
 *	the lambda parameter specifies the relaxation, convergence is guaranteed as long
 *	as 0.0 < lambda < 2.0 and goes to zero eventually.
 */
int relax_kacz(double *A, int m, int n, int stride, double *b, double *x0, int rowi, double lambda);

/*
 *	gradient descent iteration (method of steepest descent)
 *
 *	notice that the b-vector is not passed in explicitly, instead it is part
 *	of the initial residual res, which needs to be computed with relax_maxres
 *	before the first invocation of relax_steepdesc for the current problem.
 *
 *	ares must be an n-vector, it is used to store the matrix-vector product A*res
 *	and is required to be non-NULL by the routine.
 *
 *	If the caller is worried about drift due to roundoff, it is ok to recompute
 *	res at any time using relax_maxres.
 */
double relax_graddesc(double *A, int m, int n, int stride, double *x0, double *res, double *ares);

/*
 *	conjugate gradient iteration
 *
 *	notice that the b-vector is not passed in explicitly, instead it is part
 *	of the initial residual res, which needs to be computed with relax_maxres
 *	before the first invocation of relax_steepdesc for the current problem.
 *
 *	dir must be an n-vector, and its contents must be identical to res on first
 *	invocation. the routine will update the contents of dir to be what the current
 *	line search direction is, which will differ from the residual on subsequent
 *	iterations.
 *
 *	adir must be an n-vector, it is used to store the matrix-vector product A*dir
 *	and is required to be non-NULL by the routine.
 *
 *	rlen2 must be a pointer to a single double variable. it needs to be initialized
 *	to relax_dot(res, 1, res, 1, n) on the first pass, and is used to avoid recomputing
 *	it in the beginning of every iteration.
 *
 *	If the caller is worried about drift due to roundoff, it is acceptable to
 *	recompute res between invocations using relax_maxres.
 */
double relax_conjgrad_init(double *A, int m, int n, int stride, double *x, double *b, double *res, double *dir, double *rlen2);
double relax_conjgrad(double *A, int m, int n, int stride, double *x0, double *res, double *dir, double *tmp, double *rlen2);

double relax_cgls_init(double *A, int m, int n, int stride, double *x, double *b, double *res, double *dir, double *reslen2);
double relax_cgls(double *A, int m, int n, int stride, double *x0, double *res, double *dir, double *tmp, double *reslen2);

double relax_cgmn_init(double *A, int m, int n, int stride, double *x, double *b, double *res, double *dir, double *reslen2);
double relax_cgmn(double *A, int m, int n, int stride, double *x0, double *res, double *dir, double *tmp, double *tmp2, double *reslen2);

/*
 *	relax_solve and relax_gauss destructively solves the system
 *	Ax = b for x.
 *
 *	After returning, A contains a diagonal matrix and b contains
 * 	the vector x.
 *
 *	If the matrix is found to be (too close to) singular, relax_solve
 *	and relax_gauss return -1.
 */
int relax_solve(double *A, int m, int n, int stride, double *b);
int relax_gauss(double *A, int m, int n, int stride, double *b);

/*
 *	relax_svd computes the singular value decomposition UΣV (not UΣVᵀ) of a matrix
 *	parameters behave as follows
 *
 *	- U is the input matrix, and the left orthogonal transformation matrix U
 *	- m is the number of rows
 *	- n is the number of columns
 *	- ustride the row offset for matrix u (often n)
 *	- V is the right orthogonal transformation matrix
 *	- vstride is the row offset for matrix v (often n)
 *	- w is the n-vector of singular values (diagonal of Σ)
 *	- tmpvec is an n-vector used for temporary storage by the algorithm
 *
 *	For an under-determined system (m < n), you should call relax_svd on Aᵀ
 *	to obtain Aᵀ = UΣVᵀ instead, which can be transposed back into A = VΣUᵀ,
 *	Σ being its own transpose.
 *
 *	returns 0 on success, -1 if the algorithm didn't converge after 30 iterations.
 */
int relax_svd(double *U, int m, int n, int ustride, double *V, int vstride, double *w, double *tmpvec);

/*
 *	relax_pinvb relax_pinvtb computes the Moore-Penrose pseudoinverse from a
 *	singular value decomposition, knowing that if A = UΣV, then
 *	A⁺ = VᵀΣ⁻¹U is its pseudoinverse.
 *
 *	The routine intentionally doesn't call the SVD itself, because the
 *	user typically needs to manipulate the singular values to his liking.
 *	This kind of use is best supported by having the user call relax_svd
 *	himself, do the necessary manipulations (such as eliminating values
 *	very close to zero) and call relax_pinv.
 *
 *	U is MxN (has m rows and n columns)
 *	C is NxM (has n rows and m columns)
 */
void relax_pinvb(double *U, int m, int n, int ustride, double *V, int vstride, double *w, double *b, double *x, double *tmp);
void relax_pinvtb(double *U, int m, int n, int ustride, double *V, int vstride, double *w, double *b, double *x, double *tmp);

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
 *	relax_ab and relax_atb multiply b by A (transpose of A) and store the result in c
 *	A is MxN (has m rows and n columns)
 *	b is Nx1 (has m rows for relax_ab, n for relax_atb)
 *	c is Mx1 (has n rows for relax_ab, m for relax_atb)
 *
 *	relax_at stores the transpose of A into C
 *	A is MxN (has m rows and n columns)
 *	C is NxM (has n rows and m columns)
 */
void relax_ata(double *A, int m, int n, int astride, double *C, int cstride);
void relax_aat(double *A, int m, int n, int astride, double *C, int cstride);
void relax_atb(double *A, int m, int n, int astride, double *b, double *c);
void relax_at(double *A, int m, int n, int astride, double *C, int cstride);
void relax_ab(double *A, int m, int n, int astride, double *b, double *c);

/*
 *	Relax_maxres computes the residual vector b - Ax for a system Ax = b
 *	and returns the maximum absolute value in that vector.
 */
double relax_maxres(double *A, int m, int n, int stride, double *x, double *b, double *res);

/*
 *	relax_dot computes the dot product between the vectors a and b.
 *
 *	the stride parameters can be used to use a matrix column as a vector,
 *	but should be passed in as 1 for row vectors or when using a matrix row.
 */
double relax_dot(double *a, int astride, double *b, int bstride, int n);

/*
 *	relax_bypax computes y = beta*y + alpha*x for scalar alpha and n-vectors y and x.
 *
 *	the stride parameters can be used to use a matrix column as a vector,
 *	but should be passed in as 1 for row vectors or when using a matrix row.
 */
void relax_bypax(double beta, double *y, int ystride, double alpha, double *x, int xstride, int n);

/*
 *	relax_copy copies the n-vector x to the n-vector y
 *
 *	as for other routines, the stride parameter is generally 1, but other values
 *	can be used to use non-consecutive linear patterns like matrix columns as vectors.
 */
void relax_copy(double *y, int ystride, double *x, int xstride, int n);
