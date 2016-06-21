# Relaxation step for dense and sparse M-by-N matrices

[![Build Status](https://travis-ci.org/aki5/librelax.svg?branch=master)](https://travis-ci.org/aki5/librelax)


```
double relax_dense(
	double *A, int m, int n, int stride,
	double *b,
	double *x0, double *x1, double *res,
	double w
);

double relax_sparse(
	double *A, int *m, int *n,  int len,
	double *b,
	double *x0, double *x1, double *res,
	double w
);
```

There are M rows and N columns in the M-by-N matrix A, and the vectors b, x0 and x1 are column vectors of N elements each. In the source code, i indicates the row (0 to m-1), and j indicates the column (0 to n-1).

The dense version assumes the array A has the matrix elements stored in a dense fashion, rows of N columns each being offset by stride elements from each other. The  stride parameter can be used to pass in a part of a larger matrix without copying its contents.

The sparse version has a densely packed array of len elements, and for each element there are explicit row and column numbers in the M and N arrays respectively, eg. element A[k] has row m[k] and column n[k].

If x0 and x1  are the same pointer, the step is gauss-seidel type (in-place update), but if they are different, the step is jacobi type (read from x0, write to x1)

Relaxation factor w of 1.0 results in classical jacobi (x0 != x1) or gauss-seidel (x0 == x1), while w < 1.0 is under-relaxed and w > 1.0 is over-relaxed.

Res is used to store a residual vector (which is computed as a side effect).

Return value is the maximum absolute value in the residual vector.
