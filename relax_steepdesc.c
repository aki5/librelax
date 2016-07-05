#include <stddef.h>
#include <math.h>
#include "relax.h"

/*
 *	The Gradient Descent iteration solves Ax = b for x,
 *	it is also known as the method of steepest descent.
 *
 *
 *	Instead of solving Ax = b directly, we'll reformulate the problem as
 *	finding the minimum of its integral F(x) instead.
 *
 *	Assuming symmetric A, we have
 *
 *		f(x) = Ax - b = 0.5*(A+Aᵀ)x - b
 *
 *	which can be taken as f(x) = δF(x)/δx and integrated into
 *
 *		F(x) = 0.5*xᵀAx - xᵀb.
 *
 *	And now we have both, the function F(x) we want to minimize and its
 *	gradient f(x) readily available.
 *
 *	We know from basic analysis results that the gradient vector points
 *	to the direction where the function values increase the quickest, which
 *	we might call steepest ascent. We can thus say that the direction of
 *	steepest descent is the opposite of that, ie.
 *
 *		-f(x) = b - Ax
 *
 *	With these formulas, the method of steepst descent is very simple: just
 *	move the solution to the direction of the steepest descent by the right
 *	distance, and we should always get closer to the solution. The trick is
 *	of course in determining how far to move in that direction.
 *
 *	Intuitively, a good choice is to move exactly the distance that
 *	minimizes F(x) along the line defined by the direction of steepest
 *	descent.
 *
 *	Doing a line search for that point is again based on knowing that the
 *	minimum is where the gradient is zero, but along the line defined by the
 *	current gradient instead of the more usual x-axis. This transforms the
 *	equation into the dot product of the two gradients being zero (ie.
 *	length of the projection of one to the other is zero, ie. that they are
 *	orthogonal). If we define r = -f(x), it quite easily follows that
 *
 *		r₁ᵀr₀ = 0
 *	<=>	(b - Ax₁)ᵀr₀ = 0
 *	<=>	(b - A(x₀+a₀r₀))ᵀr₀ = 0
 *	<=>	(b - Ax₀)ᵀr₀ - a₀(Ar₀)ᵀr₀ = 0
 *	<=>	(b - Ax₀)ᵀr₀ = a₀(Ar₀)ᵀr₀
 *	<=>	r₀ᵀr₀ = a₀r₀ᵀ(Ar₀)
 *	<=>
 *		a₀ = r₀ᵀr₀ / r₀ᵀAr₀
 *
 *	and so we have
 *
 *		a₀ = r₀ᵀr₀ / r₀ᵀAr₀
 *		x₁ = x₀ + a₀r₀
 *
 *	to get a better solution x₁, and
 *
 *		x₁ = x₀ + a₀r₀
 *	<=>	-Ax₁ = -Ax₀ - A(a₀r₀)
 *	<=>	-Ax₁ + b = -Ax₀ - A(a₀r₀) + b
 *	<=>	b - Ax₁ = b - Ax₀ - A(a₀r₀)
 *	<=>	r₁ = r₀ - A(a₀r₀)
 *	<=>
 *		r₁ = r₀ - a₀Ar₀
 *
 *	to get the next residual r₁.
 *
 *	This gives a better 'steepest descent' approximation x₁ and its residual
 *	when given a previous one with its corresponding residual r₀ = b₀ - Ax₀.
 *
 *	As you might have noticed, the direction of steepest descent happens to
 *	be exactly the residual value for the system Ax = b as computed by
 *	relax_maxres, which is typically computed to understand how far off a
 *	current guess is.
 *
 *	For some uses of the residual the exact direction of the vector does not
 *	matter, and so some routines may compute the residual as Ax - b instead.
 *
 *	Verify the direction of the residual vector before using an unknown
 *	routine, since it may point the exact wrong way for steepest descent.
 *	If so, it is easy to reverse, but care must be taken.
 *
 */
double
relax_graddesc(double *A, int m, int n, int stride, double *x0, double *r0, double *ar0)
{
	double a, maxres;
	int i;

	relax_ab(A, m, n, stride, r0, ar0);
	a = relax_dot(r0, 1, r0, 1, n) / relax_dot(r0, 1, ar0, 1, n);

	maxres = fabs(r0[0] - a*ar0[0]);
	for(i = 0; i < m; i++){
		x0[i] = x0[i] + a*r0[i];
		r0[i] = r0[i] - a*ar0[i];
		maxres = maxres > fabs(r0[i]) ? maxres : fabs(r0[i]);
	}

	return maxres;
}
