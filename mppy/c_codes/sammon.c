#include <stdio.h>
#include <math.h>
#include <stdlib.h>

extern void sammon(double **distance_matrix, 
	double **initial_projection,
	double **projection_aux,
	int instances, 
	int max_iter,
	double magic_factor,
	double tol){
	int count;
	double delta_pq = 0.0;
	double c = 0.0;
	double sum_dist_rn = 0.0;
	double sum_dist_r2 = 0.0;
	double sum_inder_1 = 0.0;
	double sum_inder_2 = 0.0;	
	int p, q, j;
	double x1x2, y1y2, dist_pj;
	int a, b;


	
    //printf("%lf\n", distance_matrix[1][0]);
    //printf("%lf\n",initial_projection[1][1]);
    //printf("%lf\n",projection_aux[1][1]);
    //printf("%d\n", instances);
    //printf("%d\n", max_iter);
    //printf("%lf\n", magic_factor);
    //printf("%lf\n", tol);
    

	for (int i =0; i<max_iter; i++){
		int count;
	double delta_pq = 0.0;
	double c = 0.0;
	double sum_dist_rn = 0.0;
	double sum_dist_r2 = 0.0;
	double sum_inder_1 = 0.0;
	double sum_inder_2 = 0.0;	
	int p, q, j;
	double x1x2, y1y2, dist_pj;
	int a, b;
	
		sum_dist_rn = 0.0;	
		
		for(a=0; a<instances; a++){
			for(b=0; b<instances; b++){
				if(distance_matrix[a][b] < tol){
					distance_matrix[a][b] = tol;
				}

				sum_dist_rn += distance_matrix[a][b];
			}
		}

		c = -2.0 / sum_dist_rn;
		//printf("sum_dist_rn -> %lf\n", sum_dist_rn);
		//printf("c -> %lf\n", -2.0 / sum_dist_rn);

		for(p=0; p<instances; p++){
			
			for(q=0; q<2; q++){
				sum_inder_1 = 0.0;
				sum_inder_2 = 0.0;

				for(j=0; j<instances; j++){
					if(j != p){
						x1x2 = projection_aux[p][0] - projection_aux[j][0];
						y1y2 = projection_aux[p][1] - projection_aux[j][1];
						dist_pj = sqrt(abs(x1x2 * x1x2 + y1y2 * y1y2));

						if(dist_pj < tol){
							dist_pj = tol;
						}

						double d = ((distance_matrix[p][j] - dist_pj) / (distance_matrix[p][j] * dist_pj));
						double e = initial_projection[p][q] - initial_projection[j][q];
						sum_inder_1 += d * e;
						
						double f = (1 / (distance_matrix[p][j] * dist_pj));
						double g = ((distance_matrix[p][j] - dist_pj) - ((pow((initial_projection[p][q] - initial_projection[j][q]),2) / dist_pj) * (1 + ((distance_matrix[p][j] - dist_pj) / dist_pj))));
						sum_inder_2 += f * g;

					}
				}

				delta_pq = ((c * sum_inder_1) / abs(c * sum_inder_2));
				initial_projection[p][q] -= magic_factor * delta_pq;
			}
		}
	//printf("%lf\n", initial_projection[2][1]);
	}

	//free(distance_matrix);
	//free(projection_aux);

}