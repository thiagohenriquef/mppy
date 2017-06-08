#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

extern void lsp(double **neighbors, double **A, double **b, int *sample_indices, double **sample_proj, int nc, int n_neighbors, int instances, float weight){
	int i = 0;
	int j = 0;

	for(i=0; i<instances; i++){
		A[i][i] = 1.0;
		
		for(int j=0; j<n_neighbors; j++){
			int value = neighbors[i][j];
			A[i][value] = (-(1.0 / n_neighbors));
		}
	}

	for (i =0; i < nc; i++){
		A[i+instances][sample_indices[i]] = 1.0;
	}

	for (j=0; j<nc; j++){
		b[j+instances][0] = sample_proj[j][0] * weight;
		b[j+instances][1] = sample_proj[j][1] * weight;
	}


	//matrix_2d = np.linalg.lstsq(A,b)
}