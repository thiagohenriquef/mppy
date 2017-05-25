#include<stdio.h>
#include<stdlib.h>
#include<math.h>

extern void neighborhood_hit(double **distance_r2, double *values, double *clusters, int instances, int max_neighbors){
    int n, i;
    int total = 0;
    double c;

    for(n=0; n<max_neighbors; n++){
        double percentage = 0.0;

        for(i=0; i<instances; i++){
            c = clusters[i];

            for(int j=0; j<n+1; j++){
                if (c==distance_r2[i][j]){
                    total++;
                }
            }
            percentage += total / (n+1);
        }
        values[n] = percentage / instances;
    } 
}
