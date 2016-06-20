# Relaxation step for a dense M-by-N matrix

```
double relax_step(double *A, int m, int n, int stride, double *b, double *x0, double *x1, double *res, double w);
```

There are M rows and N columns in the M-by-N matrix A.
In the source code, i indicates the row (0 to m-1), and j indicates the column (0 to n-1).

If x0 and x1 vectors are the same, the step is gauss-seidel type (overwrite in-place), but
if x0 and x1 vectors are different, the step is jacobi type (non destructive update)

Relaxation factor w of 1.0 results in classic jacobi or gauss-seidel, while w < 1.0
is under-relaxed, w > 1.0 is over-relaxed.

Res is used to store a residual vector (which is computed as a side effect).

Return value is the maximum absolute residual in the residual vector.

