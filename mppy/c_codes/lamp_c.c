#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
//#include "defs_and_types.h"
//#include "svd.h"

typedef enum { false, true } bool;

#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : - fabs(a))

static double maxarg1,maxarg2;
#define FMAX(a,b) (maxarg1 = (a),maxarg2 = (b),(maxarg1) > (maxarg2) ? (maxarg1) : (maxarg2))

static int iminarg1,iminarg2;
#define IMIN(a,b) (iminarg1 = (a),iminarg2 = (b),(iminarg1 < (iminarg2) ? (iminarg1) : iminarg2))

static double sqrarg;
#define SQR(a) ((sqrarg = (a)) == 0.0 ? 0.0 : sqrarg * sqrarg)

int svdcmp(double **a, int nRows, int nCols, double *w, double **v);

// prints an arbitrary size matrix to the standard output
void printMatrix(double **a, int rows, int cols);
void printMatrix(double **a, int rows, int cols) {
	int i,j;

	for(i=0;i<rows;i++) {
		for(j=0;j<cols;j++) {
			printf("%.4lf ",a[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

// prints an arbitrary size vector to the standard output
void printVector(double *v, int size);
void printVector(double *v, int size) {
	int i;

	for(i=0;i<size;i++) {
		printf("%.4lf ",v[i]);
	}
	printf("\n\n");
}

// calculates sqrt( a^2 + b^2 ) with decent precision
double pythag(double a, double b);
double pythag(double a, double b) {
	double absa,absb;

	absa = fabs(a);
	absb = fabs(b);

	if(absa > absb)
		return(absa * sqrt(1.0 + SQR(absb/absa)));
	else
		return(absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}

/*
  Modified from Numerical Recipes in C
  Given a matrix a[nRows][nCols], svdcmp() computes its singular value 
  decomposition, A = U * W * Vt.  A is replaced by U when svdcmp 
  returns.  The diagonal matrix W is output as a vector w[nCols].
  V (not V transpose) is output as the matrix V[nCols][nCols].
*/
int svdcmp(double **a, int nRows, int nCols, double *w, double **v) {
	int flag,i,its,j,jj,k,l,nm;
	double anorm,c,f,g,h,s,scale,x,y,z,*rv1;
	rv1 = malloc(sizeof(double)*nCols);
	if(rv1 == NULL) {
		printf("svdcmp(): Unable to allocate vector\n");
		return(-1);
	}
	printf("memoria alocada\n");

	g = scale = anorm = 0.0;
	for(i=0;i<nCols;i++) {
		printf("primerio loop svd\n");
		l = i+1;
		rv1[i] = scale*g;
		g = s = scale = 0.0;
		if(i < nRows) {
			printf("if svd\n");
			for(k=i;k<nRows;k++){ 
					scale += fabs(a[k][i]);
				}
				printf("loop do loop\n");
				if(scale) {
					for(k=i;k<nRows;k++) {
						a[k][i] /= scale;
						s += a[k][i] * a[k][i];
					}
					f = a[i][i];
					g = -SIGN(sqrt(s),f);
					h = f * g - s;
					a[i][i] = f - g;
					for(j=l;j<nCols;j++) {
						for(s=0.0,k=i;k<nRows;k++) s += a[k][i] * a[k][j];
							f = s / h;
						for(k=i;k<nRows;k++) a[k][j] += f * a[k][i];
					}
				for(k=i;k<nRows;k++) a[k][i] *= scale;
			}
	}
	printf("saiu primeiro loop svd\n");
	w[i] = scale * g;
	g = s = scale = 0.0;
	if(i < nRows && i != nCols-1) {
		for(k=l;k<nCols;k++) scale += fabs(a[i][k]);
			if(scale)  {
				for(k=l;k<nCols;k++) {
					a[i][k] /= scale;
					s += a[i][k] * a[i][k];
				}
				f = a[i][l];
				g = - SIGN(sqrt(s),f);
				h = f * g - s;
				a[i][l] = f - g;
				for(k=l;k<nCols;k++) rv1[k] = a[i][k] / h;
					for(j=l;j<nRows;j++) {
						for(s=0.0,k=l;k<nCols;k++) s += a[j][k] * a[i][k];
							for(k=l;k<nCols;k++) a[j][k] += s * rv1[k];
						}
					for(k=l;k<nCols;k++) a[i][k] *= scale;
				}
		}
		anorm = FMAX(anorm, (fabs(w[i]) + fabs(rv1[i])));

		printf(".");
		fflush(stdout);
	}

	for(i=nCols-1;i>=0;i--) {
		if(i < nCols-1) {
			if(g) {
				for(j=l;j<nCols;j++)
					v[j][i] = (a[i][j] / a[i][l]) / g;
				for(j=l;j<nCols;j++) {
					for(s=0.0,k=l;k<nCols;k++) s += a[i][k] * v[k][j];
						for(k=l;k<nCols;k++) v[k][j] += s * v[k][i];
					}
			}
			for(j=l;j<nCols;j++) v[i][j] = v[j][i] = 0.0;
		}
	v[i][i] = 1.0;
	g = rv1[i];
	l = i;
	printf(".");
	fflush(stdout);
}

for(i=IMIN(nRows,nCols) - 1;i >= 0;i--) {
	l = i + 1;
	g = w[i];
	for(j=l;j<nCols;j++) a[i][j] = 0.0;
		if(g) {
			g = 1.0 / g;
			for(j=l;j<nCols;j++) {
				for(s=0.0,k=l;k<nRows;k++) s += a[k][i] * a[k][j];
					f = (s / a[i][i]) * g;
				for(k=i;k<nRows;k++) a[k][j] += f * a[k][i];
			}
		for(j=i;j<nRows;j++) a[j][i] *= g;
	}
else
	for(j=i;j<nRows;j++) a[j][i] = 0.0;
		++a[i][i];
	printf(".");
	fflush(stdout);
}

for(k=nCols-1;k>=0;k--) {
	for(its=0;its<30;its++) {
		flag = 1;
		for(l=k;l>=0;l--) {
			nm = l-1;
			if((fabs(rv1[l]) + anorm) == anorm) {
				flag =  0;
				break;
			}
			if((fabs(w[nm]) + anorm) == anorm) break;
		}
		if(flag) {
			c = 0.0;
			s = 1.0;
			for(i=l;i<=k;i++) {
				f = s * rv1[i];
				rv1[i] = c * rv1[i];
				if((fabs(f) + anorm) == anorm) break;
				g = w[i];
				h = pythag(f,g);
				w[i] = h;
				h = 1.0 / h;
				c = g * h;
				s = -f * h;
				for(j=0;j<nRows;j++) {
					y = a[j][nm];
					z = a[j][i];
					a[j][nm] = y * c + z * s;
					a[j][i] = z * c - y * s;
				}
			}
		}
		z = w[k];
		if(l == k) {
			if(z < 0.0) {
				w[k] = -z;
				for(j=0;j<nCols;j++) v[j][k] = -v[j][k];
			}
		break;
	}
	if(its == 29) printf("no convergence in 30 svdcmp iterations\n");
	x = w[l];
	nm = k-1;
	y = w[nm];
	g = rv1[nm];
	h = rv1[k];
	f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
	g = pythag(f,1.0);
	f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g,f))) - h)) / x;
	c = s = 1.0;
	for(j=l;j<=nm;j++) {
		i = j+1;
		g = rv1[i];
		y = w[i];
		h = s * g;
		g = c * g;
		z = pythag(f,h);
		rv1[j] = z;
		c = f/z;
		s = h/z;
		f = x * c + g * s;
		g = g * c - x * s;
		h = y * s;
		y *= c;
		for(jj=0;jj<nCols;jj++) {
			x = v[jj][j];
			z = v[jj][i];
			v[jj][j] = x * c + z * s;
			v[jj][i] = z * c - x * s;
		}
		z = pythag(f,h);
		w[j] = z;
		if(z) {
			z = 1.0 / z;
			c = f * z;
			s = h * z;
		}
		f = c * g + s * y;
		x = c * y - s * g;
		for(jj=0;jj < nRows;jj++) {
			y = a[jj][j];
			z = a[jj][i];
			a[jj][j] = y * c + z * s;
			a[jj][i] = z * c - y * s;
		}
	}
	rv1[l] = 0.0;
	rv1[k] = f;
	w[k] = x;
}
printf(".");
fflush(stdout);
}
printf("\n");

free(rv1);

return(0);
}


extern void lamp(double **data_matrix,
	double **matrix_2d,
	double **AtB,
	double **sample_data,
	double **sample_proj,
	int instances,
	int d,
	int k, 
	int r,
	int n
	){

	int columns = d;
	int p = 0;
	int i;
	double p_star[d];
	memset(p_star, 0, d);
	double q_star[r];
	memset(q_star, 0, r);
	double local_w[n];
	memset(local_w, 0, n);
	double neighbors_index[n];
	memset(neighbors_index, 0, n);
	//printf("entrou aqui\n");

	for(p; p<instances; p++){
		
		double X[columns];
		for(int count_2=0; count_2<columns; count_2++){
			X[count_2] = data_matrix[p][count_2];
		}

		double p_sum[d];
		memset(p_sum, 0, d);
		double q_sum[r];
		memset(q_sum, 0, r);
		double w_sum = 0.0;
		bool jump = false;

		double local_w[n];
		memset(local_w, (-1.0/0.0), n);
		//printf("e aqui?\n");
		for(i=0; i<k; i++){
			double P[columns];
			for(int cc=0; cc<columns; cc++){
				P[cc] = sample_data[i][cc];
			}
			double Q[2];
			Q[0] = sample_proj[i][0];
			Q[1] = sample_proj[i][1];

			double w = 0.0;
			int j = 0;

			for(int j=0; j<d;j++){
				w += (X[j] - P[j]) * (X[j] - P[j]);
			}

			if (w < 0.000001){
				matrix_2d[p][0] = Q[0];
				matrix_2d[p][1] = Q[1];
				jump = true;
				break;
			}
			//printf("linha 361\n");

			if (w < local_w[n-1]){
				//printf("entrou no if\n");
				for(int j; j<n; j++){
					//printf("entrou no for\n");
					if(local_w[j] > w){
						for(int m=n-1; m>j; m--){
							local_w[m] = local_w[m-1];
							neighbors_index[m] = neighbors_index[m - 1];
						}
						local_w[j] = w;
						neighbors_index[j] = i;
						//printf("%d\n", i);
						break;
					}
				}
			}
        	//printf("passou\n");
			//printf("%d\n", p);
			//printf("%d\n", i);
			//printf("%d\n", k);

		}
		if (jump==true){
			continue;
		}
		//printf("entrou aqui\n");

		for(i=0; i<n; i++){
			//printf("aqui??\n");
			double P[columns];
			for(int cc=0; cc<columns; cc++){
				P[cc] = sample_data[(int) (neighbors_index[i])][cc];
			}
			double Q[2];
			Q[0] = sample_proj[(int) neighbors_index[i]][0];
			Q[1] = sample_proj[(int) (neighbors_index[i])][1]; 
			//printf("%lf\n", Q[1]);       	

			local_w[i] = 1.0 / local_w[i];

			for(int j=0; j<d; j++){
				p_sum[j] = p_sum[j] + P[j] * local_w[i];
			}
			q_sum[0] = q_sum[0] + Q[0] * local_w[i];
			q_sum[1] = q_sum[1] + Q[1] * local_w[i];

			w_sum = w_sum + local_w[i];
		}
		printf("meio passo 1\n");
		for (int j=0; j < d; j++){
			p_star[j] = p_sum[j] / w_sum;
		}

		q_star[0] = q_sum[0] / w_sum;
		q_star[1] = q_sum[1] / w_sum;


        //STEP 2
		printf("incio paso2\n");
		int j;
		for(i=0; i<d; i++){
			double x = 0.0;
			double y = 0.0;

			for(j=0;j<n;j++){
				double P[columns];
				for(int cc=0; cc<columns; cc++){
					P[cc] = sample_data[(int) (neighbors_index[j])][cc];
				}
				double Q[2];
				Q[0] = sample_proj[(int) neighbors_index[j]][0];
				Q[1] = sample_proj[(int) (neighbors_index[j])][1];   

				double w_sqrt = sqrt(abs(local_w[j]));

				double a_ij = (P[i] - p_star[i]) * w_sqrt;

				x = x + (a_ij * ((Q[0] - q_star[0]) * w_sqrt));
				y = y + (a_ij * ((Q[1] - q_star[1]) * w_sqrt));
			}
			AtB[i][0] = x;
			AtB[i][1] = y;
			printf("final passo 2\n");
		}

        //STEP 3
		double **U;
        //memset(U, 0, sizeof(U[0][0]) * instances * columns);
		U = (double **)malloc(sizeof(double[instances][2]));

		printf("depois do U\n");
		double s[2];
		memset(s, 0, 2);
		printf("depois do s\n");
		double **V;
        //memset(V, 0, sizeof(V[0][0]) * columns * columns);
		V = (double **)malloc(sizeof(double[2][2]));
		printf("depois do V\n");

        //svd(instances, 2, AtB, U, s, V);
		int alea = svdcmp(AtB, instances, 2, s, V);

		printf("chamou svd\n");
		double v_00 = V[0][0];
		double v_01 = V[0][1];
		double v_10 = V[1][0];
		double v_11 = V[1][1];
		printf("%lf\n", V[1][1]);

		double x = 0.0;
		double y = 0.0;
		j = 0;

		for(j=0; j<d; j++){
			double diff = X[j] - p_star[j];
			double u_j0 = U[i][0];
			double u_j1 = U[i][1];

			x += diff * (u_j0 * v_00 * u_j1 * v_01);
			y += diff * (u_j0 * v_10 * u_j1 * v_11);
		}

		x = x + q_star[0];
		y = y + q_star[1];

		matrix_2d[p][0] = x;
		matrix_2d[p][1] = y;
	}

}
