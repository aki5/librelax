# A few simple routines for dense M-by-N matrices

[![Build Status](https://travis-ci.org/aki5/librelax.svg?branch=master)](https://travis-ci.org/aki5/librelax)

## Coordinate Descent (aka. Gauss-Seidel) method

Relax_coordesc implements a relaxation step for a dense M-by-N matrix

The method will converge if A is symmetric and positive (semi)definite. Notably, the normal equations AᵀA and AAᵀ which can be used to make least squares and least norm solutions for over- and underdetermined systems fall into this category for any matrix.

```
double relax_coordesc(
	double *A, int m, int n, int stride,
	double *b,
	double *x0,
	double *res
);
```

There are M rows and N columns in the M-by-N matrix A, and the vectors b and x0 are column vectors of N elements each. In the source code, i indicates the row (0 to m-1), and j indicates the column (0 to n-1).

Elements of A need to be stored in a dense matrix format, with M rows of N columns. Rows are offset from each other by stride elements. The  stride parameter can be used to pass in a part of a larger matrix without copying its contents, but is often passed in as N.

The step makes an in-place update to the vector x0.

Res is used to store a residual vector (which is computed as a side effect).

Return value is the maximum absolute value in the residual vector.

## Direct method solver

Relax_solve destructively soves the system Ax = b for x. After relax_solve returns, A contains a diagonal matrix of ones and b contains the vector x.
The implementation is a simple gauss-jordan elimination, provided as a baseline for testing and other purposes.

```
int relax_solve(double *A, int m, int n, int stride, double *b);
```

Return value is 0 on success, but if the matrix is found to be singular, relax_solve returns -1.

## Least Squares and Minimum Norm utilities

When there are too many equations `(M > N)`, it is possible to reduce the number of rows to N by multiplying both, the matrix A and vector b by Aᵀ from the left. Solving the resulting NxN system `Aᵀ*A*x = Aᵀ*b` for x results in a least squares fit to the original overdetermined system.

When there are too few equations `(M < N)`, it is similarly possible to reach a solution by using Aᵀ. The MxM system `A*Aᵀ*w = b` can be solved for w from which the minimum norm fit is acquired by computing `x = Aᵀ*w`.

The least squares claim follow fairly simply from taking a gradient of the squared norm of residual of the system, and then noticing that a positive square function has its minimum value when all derivatives are zero. The minimum norm solution minimizes the norm of the result instead of that of the residual, and can be derived using the same basic insights.

The solution could also be expressed in matrix form, `x = A⁺ * b`, where `A⁺ = (Aᵀ * A)⁻¹ * Aᵀ` is known as the Moore-Penrose pseudoinverse.

```
void relax_aat(
	double *A, int m, int n, int astride,
	double *C, int cstride
);

void relax_ata(
	double *A, int m, int n, int astride,
	double *C, int cstride
);

void relax_atb(
	double *A, int m, int n, int astride,
	double *b, double *c
);

void relax_ab(
	double *A, int m, int n, int astride,
	double *b, double *c
);
```

relax_aat computes Aᵀ*A (transpose of A by A) and stores the result in C, where A is MxN (has m rows and n columns) and C is NxN (has n rows and n columns)

relax_ata multiplies A*Aᵀ (A by transpose of A) and stores the result in C, where A is MxN (has m rows and n columns) and C is NxN (has n rows and n columns)

relax_ab and relax_atb multiplies b by A (Aᵀ for relax_atb) and stores the result in c.
