# Relaxation step for dense and sparse M-by-N matrices

[![Build Status](https://travis-ci.org/aki5/librelax.svg?branch=master)](https://travis-ci.org/aki5/librelax)

## Successive over-relaxation step

```
double relax_sor(
	double *A, int m, int n, int stride,
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

## Least squares formulation 

```
void relax_ata(double *A, int m, int n, int astride, double *C, int cstride);
void relax_atb(double *A, int m, int n, int astride, double *b, double *c);
```

When it turns out that we have too many equations (M > N), it is possible to reduce the number of rows to N by multiplying both, the matrix A and vector b by the transpose of A.

Solving the resulting NxN system for x results in a least squares fit to the original overdetermined system.

This outrageous claim follows fairly simply from taking a gradient of the squared length of residual of the system, and then noticing that a positive square function has its minimum value when all derivatives are zero.

The solution could also be expressed in matrix form, x = A+ * b, where A+ == (At*A)^-1 * At is also known as Moore-Penrose pseudoinverse.

relax_ata multiplies A by its own transpose and store the result in C, where A is MxN (has m rows and n columns) and C is NxN (has n rows and n columns)

relax_atb multiplies b by transpose of A, store the result in c, where A is MxN (has m rows and n columns) and c is Nx1 (has n rows).
