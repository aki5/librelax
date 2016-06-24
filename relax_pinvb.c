#include <math.h>
#include "relax.h"
/*
 *	Relax_pinvb solves Ax = b using values from a previous call to relax_svd(A),
 *	and produces a least squares fit for an over-determined system,
 *	ie. len(b) > len(x)
 *
 *		A  = UΣVᵀ
 *		=> A⁺ = V Σ⁻¹Uᵀ
 *		=> x = V *Σ⁻¹*Uᵀ*b
 */
void
relax_pinvb(double *U, int m, int n, int ustride, double *V, int vstride, double *w, double *b, double *x, double *tmp)
{
	int i;

	// tmp = Uᵀ*b
	relax_atb(U, m, n, ustride, b, tmp);
	// tmp = Σ⁻¹*tmp
	for(i = 0; i < n; i++)
		if(fabs(w[i]) > 1e-12)
			tmp[i] = tmp[i] / w[i];
	// x = V*tmp
	relax_ab(V, n, n, vstride, tmp, x);
}
