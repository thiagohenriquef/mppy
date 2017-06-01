#include<stdio.h>
#include<stdlib.h>
#include<math.h>

extern double* neighborhood_hit(double **distance_r2, double *clusters, int instances, int max_neighbors){
    double *values;
    values = (double *)malloc(max_neighbors *sizeof(double));
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

    return values;
}

extern double* neighborhood_hit(double **distance_r2, double *clusters, int instances, int max_neighbors){
    double *values;
    values = (double *)malloc(max_neighbors *sizeof(double));
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

    return values;
}