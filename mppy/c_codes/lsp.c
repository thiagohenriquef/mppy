#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void lsp(int **Dx, double **A, double **b, double **matrix_2d, int *sample_indices, double **sample_proj, int nc, int n_neighbors, int instances){
	int i = 0;
	int j = 0;

	for(i=0; i<instances; i++){
		A[i][i] = 0.0;
		for(int j=0; j<instances; j++){
			A[i][Dx[i][j]] = (-(1.0 / n_neighbors));
		}
	}

	int count = 0;
	for (i = instances; i < nc; i++){
		A[i][sample_indices[count]] = 1.0;
		count++;
	}

	for (j=instances; j<nc; j++){
		b[j+instances][0] = sample_proj[j][0];
		b[j+instances][1] = sample_proj[j][1];
	}


	//matrix_2d = np.linalg.lstsq(A,b)
}