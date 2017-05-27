#include<stdio.h>
#include<stdlib.h>
#include<math.h>

extern void normalized_kruskal(double **distance_rn,double **distance_r2, int instances){
    double max_rn = -INFINITY;
    double max_r2 = -INFINITY;
    int x,y;
    double value_rn, value_r2;
    double dist_rn, dist_r2;

    for(x=0;x<instances;x++){
        for(y=x+1;y<instances;y++){
            value_rn = distance_rn[x][y];
            value_r2 = distance_r2[x][y];

            if (value_r2 > max_r2){
                max_r2 = value_r2;
            }

            if (value_rn > max_rn){
                max_rn = value_rn;
            }
        }
    }

    double num = 0.0;
    double den = 0.0;

    for(x=0;x<instances;x++){
        for(y=x+1;y<instances;y++){
            dist_rn = distance_rn[x][y] / max_rn;
            dist_r2 = distance_r2[x][y] / max_r2;

            num = num + ((dist_rn - dist_r2) * (dist_rn - dist_r2));
            den = den + (dist_rn * dist_rn);
        }
    }

    printf("Stress: %lf\n", num / den);
}

extern void kruskal_stress(double **distance_rn,double **distance_r2, int instances){
    double num = 0.0;
    double den = 0.0;
    int i, j;

    for(i=0; i<instances; i++){
        for(j=1; j<instances; j++){
            double dist_rn = distance_rn[i][j];
            double dist_r2 = distance_r2[i][j];

            num += (dist_rn - dist_r2) * (dist_rn - dist_r2);
            den += dist_rn * dist_rn;
        }
    }

    double result = sqrt(num/den);
    printf("Stress: %lf\n", result);
}