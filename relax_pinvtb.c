#include <math.h>
#include "relax.h"
/*
 *	Relax_pinvtb solves Ax = b using values from a previous call to relax_svd(Aᵀ)
 *	and produces the minimum norm solution for an under-determined system,
 *	ie. len(b) < len(x)
 *
 *		Aᵀ = UΣVᵀ
 *		=> A⁺ᵀ = V Σ⁻¹Uᵀ
 *		<=> A⁺ = UᵀΣ⁻¹V
 *		=> x = Uᵀ*Σ⁻¹*V *b
 */
void
relax_pinvtb(double *U, int m, int n, int ustride, double *V, int vstride, double *w, double *b, double *x, double *tmp)
{
	int i;

	// tmp = V*b
	relax_ab(V, n, n, vstride, b, tmp);
	// tmp = Σ⁻¹*tmp
	for(i = 0; i < n; i++)
		if(fabs(w[i]) > 1e-12)
			tmp[i] = tmp[i] / w[i];
	// x = Uᵀ*tmp
	relax_atb(U, m, n, ustride, tmp, x);
}
