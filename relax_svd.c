/*
 *	This routine is adapted from svdecomp.c in XLISP-STAT 2.1 which is
 *	a code from Numerical Recipes adapted by Luke Tierney and David Betz.
 *
 *	http://www.public.iastate.edu/~dicook/JSS/paper/code/svd.c
 *	http://svn.lirec.eu/libs/magicsquares/src/SVD.cpp
 *
 *	CHANGES:
 *	- cured some of the fortranitis infection
 *	- flat arrays instead of row-pointer arrays
 *	- no mallocs, frees
 *
 *	TODO: replace this dinosaur with new code that takes advantage
 *	of more recent developments and has a clear licensing status.
 */

#include <math.h>

#define signd(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

static double
pythag(double a, double b)
{
	double absa, absb;
	absa = fabs(a);
	absb = fabs(b);
	if(absa > absb){
		return absa * sqrt(1.0 + (absb / absa) * (absb / absa));
	} else {
		return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + (absa / absb) * (absa / absb)));
	}
}

int
relax_svd(double *U, int m, int n, int ustride, double *V, int vstride, double *w, double *rv1)
{
	int flag, i, its, j, o, k, l, nm;
	double anorm, c, f, g, h, s, scale, x, y, z;

	if(m < n){
		// You should call SVD on Aᵀ to obtain Aᵀ = UWVᵀ instead,
		// which can be transposed into A = VWUᵀ
		return -1;
	}

	/* Householder reduction to bidiagonal form */
	g = scale = anorm = 0.0;
	for(i = 0; i < n; i++){
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if(i < m){
			for(k = i; k < m; k++)
				scale += fabs(U[k*ustride+i]);
			if(scale != 0.0){
				for(k = i; k < m; k++){
					U[k*ustride+i] /= scale;
					s += U[k*ustride+i] * U[k*ustride+i];
				}
				f = U[i*ustride+i];
				g = -signd(sqrt(s), f);
				h = f * g - s;
				U[i*ustride+i] = f - g;
				for(j = l; j < n; j++){
					for(s = 0.0, k = i; k < m; k++)
						s += U[k*ustride+i] * U[k*ustride+j];
					f = s / h;
					for(k = i; k < m; k++)
						U[k*ustride+j] += f * U[k*ustride+i];
				}
				for(k = i; k < m; k++)
					U[k*ustride+i] *= scale;
			}
		}
		w[i] = scale * g;
		/* right-hand reduction */
		g = s = scale = 0.0;
		if(i < m && i != n-1){
			for(k = l; k < n; k++)
				scale += fabs(U[i*ustride+k]);
			if(scale){
				for(k = l; k < n; k++){
					U[i*ustride+k] /= scale;
					s += U[i*ustride+k] * U[i*ustride+k];
				}
				f = U[i*ustride+l];
				g = -signd(sqrt(s), f);
				h = f * g - s;
				U[i*ustride+l] = f - g;
				for(k = l; k < n; k++)
					rv1[k] = U[i*ustride+k] / h;
				for(j = l; j < m; j++){
					for(s = 0.0, k = l; k < n; k++)
						s += U[j*ustride+k] * U[i*ustride+k];
					for(k = l; k < n; k++)
						U[j*ustride+k] += s * rv1[k];
				}
				for(k = l; k < n; k++)
					U[i*ustride+k] *= scale;
			}
		}
		double tmp = fabs(w[i]) + fabs(rv1[i]);
		anorm = anorm > tmp ? anorm : tmp;
	}

	/* accumulate the right-hand transformation */
	for(i = n-1; i >= 0; i--){
		if(i < n-1){
			if(g != 0.0){
				for(j = l; j < n; j++)
					V[j*vstride+i] = (U[i*ustride+j] / U[i*ustride+l]) / g;
				for(j = l; j < n; j++){
					for(s = 0.0, k = l; k < n; k++)
						s += U[i*ustride+k] * V[k*vstride+j];
					for(k = l; k < n; k++)
						V[k*vstride+j] += s * V[k*vstride+i];
				}
			}
			for(j = l; j < n; j++)
				V[i*vstride+j] = V[j*vstride+i] = 0.0;
		}
		V[i*vstride+i] = 1.0;
		g = rv1[i];
		l = i;
	}

	/* accumulate the left-hand transformation */
	for(i = n-1; i >= 0; i--){
		l = i + 1;
		g = w[i];
		for(j = l; j < n; j++)
			U[i*ustride+j] = 0.0;
		if(g != 0.0){
			g = 1.0 / g;
			for(j = l; j < n; j++){
				for(s = 0.0, k = l; k < m; k++)
					s += U[k*ustride+i] * U[k*ustride+j];
				f = (s / U[i*ustride+i]) * g;
				for(k = i; k < m; k++)
					U[k*ustride+j] += f * U[k*ustride+i];
			}
			for(j = i; j < m; j++)
				U[j*ustride+i] *= g;
		} else {
			for(j = i; j < m; j++)
				U[j*ustride+i] = 0.0;
		}
		U[i*ustride+i] += 1.0;
	}

	/* diagonalize the bidiagonal form */
	for(k = n-1; k >= 0; k--){
		/* loop over singular values */
		for(its = 1; its <= 30; its++){
			/* loop over allowed iterations */
			flag = 1;
			for(l = k; l >= 0; l--){
				/* test for splitting */
				nm = l - 1;
				if(fabs(rv1[l]) + anorm == anorm){
					flag = 0;
					break;
				}
				if(fabs(w[nm]) + anorm == anorm)
					break;
			}
			if(flag != 0){
				c = 0.0;
				s = 1.0;
				for(i = l; i < k; i++){
					f = s * rv1[i];
					rv1[i] = c * rv1[i];
					if(fabs(f) + anorm == anorm)
						break;
					g = w[i];
					h = pythag(f, g);
					w[i] = h;
					h = 1.0 / h;
					c = g * h;
					s = -f * h;
					for(j = 1; j < m; j++){
						y = U[j*ustride+nm];
						z = U[j*ustride+i];
						U[j*ustride+nm] = y * c + z * s;
						U[j*ustride+i] = z * c - y * s;
					}
				}
			}
			z = w[k];
			if(l == k){
				/* convergence */
				if(z < 0.0){
					/* make singular value nonnegative */
					w[k] = -z;
					for(j = 0; j < n; j++)
						V[j*vstride+k] = -V[j*vstride+k];
				}
				break;
			}
			if(its == 30)
				return -1;

			/* shift from bottom 2 x 2 minor */
			x = w[l];
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = pythag(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + signd(g, f))) - h)) / x;

			/* next QR transformation */
			c = s = 1.0;
			for(j = l; j <= nm; j++){
				i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s * g;
				g = c * g;
				z = pythag(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y *= c;
				for(o = 0; o < n; o++){
					x = V[o*vstride+j];
					z = V[o*vstride+i];
					V[o*vstride+j] = x * c + z * s;
					V[o*vstride+i] = z * c - x * s;
				}
				z = pythag(f, h);
				w[j] = z;
				if(z != 0.0){
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = c * g + s * y;
				x = c * y - s * g;
				for(o = 0; o < m; o++){
					y = U[o*ustride+j];
					z = U[o*ustride+i];
					U[o*ustride+j] = y * c + z * s;
					U[o*ustride+i] = z * c - y * s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}
	return 0;
}
