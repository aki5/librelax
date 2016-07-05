#include <stddef.h>
#include <math.h>
#include "relax.h"

/*
 *	The Gauss-Seidel iteration solves Ax = b for x
 *
 *	Instead of solving Ax = b directly, we'll reformulate the problem as
 *	finding the minimum of its integral F(x) instead.
 *
 *	Assuming symmetric A, we have
 *
 *		δF(x)/δx = Ax - b = 0.5*(A+Aᵀ)x - b
 *
 *	which can be integrated into
 *
 *		F(x) = 0.5*xᵀAx - xᵀb.
 *
 *	At every iteration, a coordinate i is chosen (cyclically or at random)
 *	and x_i is adjusted to minimize the above. We can achieve this by
 *	solving the partial derivative δF/δx_i = 0 for x_i, since a quadratic
 *	polynomial will have its extremum where its derivative is zero.
 *
 *	Let's start by expanding F(x) for a simple 3x3 case, to get some
 *	understanding of its inner workings.
 *
 *	F(x) = 0.5*xᵀAx - xᵀb
 *
 *		<=>
 *	                          .-           -.   .-  -.                  .-  -.
 *	           .-        -.   | a00 a01 a02 |   | x0 |   .-        -.   | b0 |
 *	F(x) = 0.5*| x0 x1 x2 | x | a10 a11 a12 | x | x1 | - | x0 x1 x2 | x | b1 |
 *	           '-        -'   | a20 a21 a22 |   | x2 |   '-        -'   | b2 |
 *	                          '-           -'   '-  -'                  '-  -'
 *		<=>
 *                                .-                        -.
 *	           .-        -.   | a00*x0 + a01*x1 + a02*x2 |
 *	F(x) = 0.5*| x0 x1 x2 | x | a10*x0 + a11*x1 + a12*x2 | - (b0*x0 + b1*x1 + b2*x2)
 *	           '-        -'   | a20*x0 + a21*x1 + a22*x2 |
 *                                '-                        -'
 *		<=>
 *
 *	F(x) = 0.5*((x0*a00*x0 + x0*a01*x1 + x0*a02*x2) +
 *	            (x1*a10*x0 + x1*a11*x1 + x1*a12*x2) +
 *	            (x2*a20*x0 + x2*a21*x1 + x2*a22*x2)) -
 *	           (     b0*x0 +     b1*x1 +     b2*x2)
 *
 *	Every matrix entry A_ij appears multiplied by x_i and x_j, and the
 *	givens b_i are multiplied by -x_i each. It is easy to see that with
 *	respect to δF/δx_i, only the ith row and ith column are of significance:
 * 	everything else is constant with respect to changes in x_i.
 *
 *	So, by declaring
 *
 *	          i-1               N
 *	         .----            .----
 *	          \                \
 *	rowsum =   ) A_ik * x_k  +  )  A_ik * x_k
 *	          /                /
 *	         '----            '----
 *	          k=0             k=i+1
 *
 *		and
 *
 *		   i-1                  N
 *		  .----              .----
 *		  \                   \
 *	colsum =   )  A_ki * x_k  +    )  A_ki * x_k
 *		  /                   /
 *		  '----              '----
 *		   k=0                k=i+1
 *
 *	we can write
 *
 *
 *
 *		δF/δx_i = 0.5*(rowsum + colsum + 2*A_ii*x_i) - b_i = 0
 *
 *	which, when solved for for x_i, gives us
 *
 *		x_i = (b_i - 0.5*(rowsum + colsum)) / A_ii.
 *
 *	Notice that for a symmetric matrix, rowsum = colsum and the above
 *	simplifies further into
 *
 *		x_i = (b_i - rowsum) / A_ii
 *
 *	which is exactly the familiar Gauss-Seidel inner loop below.
 *
 *	A variation of this method is also known as coordinate descent, and is
 *	known to converge for Symmetric Positive (Semi-)Definite as well as
 *	diagonally dominant matrices.
 *
 *	The difference between a typical coordinate descent method and
 *	Gauss-Seidel is that Gauss-Seidel sets x_i straight to minimizing error,
 *	while coordinate descent may use anything from a fixed step size to
 *	arbitrarily complex dynamic adjustments. Gauss-Seidel can be viewed as a
 *	special case of coordinate descent.
 *
 *	Notice that there's two places in the above derivation, including the
 *	very starting point, where the matrix is assumed to be symmetric. Using
 *	the method for non-symmetric cases will require changes to both, the
 *	algorithm and how it is derived, but I hope the derivation here has shed
 *	enough light on the underlying principles to allow creating custom
 *	variants where needed.
 *
 *	The Jacobi-method is a variation of this, where all the coordinates are
 *	adjusted to their minimizing positions simultaneously. The Jacobi method
 *	tends to converge to a solution slower, and tends to diverge easier.
 *
 *	There has been a recent development of the idea, extending this
 *	algorithm to work for both, under- and overdetermined systems.
 *
 *	[3] Anna Ma, Deanna Needell, Aaditya Ramdas.
 *	    Convergence properties of the randomized extended Gauss-Seidel
 *	    and Kaczmarz methods.
 *	    http://opt-ml.org/papers/OPT2015_paper_7.pdf
 */
double
relax_coordesc(double *A, int m, int n, int stride, double *b, double *x0, double *res)
{
	double sigma, rowres, maxres;
	int i, j, irow;

	maxres = 0.0;
	irow = 0;
	for(i = 0; i < m; i++){

		sigma = 0.0;
		rowres = 0.0;
		for(j = 0; j < n; j++){
			double tmp;
			tmp = A[irow+j] * x0[j];
			rowres += tmp;
			if(j != i)
				sigma += tmp;
		}

		x0[i] = (b[i] - sigma) / A[irow+i];

		rowres = b[i] - rowres;
		rowres = fabs(rowres);
		maxres = rowres > maxres ? rowres : maxres;
		if(res != NULL)
			res[i] = rowres;

		irow = irow + stride;
	}
	return maxres;
}
