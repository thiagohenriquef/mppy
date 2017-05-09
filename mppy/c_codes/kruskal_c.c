/*
    max_rn = -math.inf
    max_r2 = -math.inf

    for x in range(distance_rn.shape[0]):
        for y in range(1, distance_rn.shape[0]):
            value_rn = distance_rn[x,y]
            value_r2 = distance_r2[x,y]

            if value_r2 > max_r2:
                max_r2 = value_r2

            if value_rn > max_rn:
                max_rn = value_rn

    num = 0.0
    den = 0.0
    for i in range(distance_rn.shape[0]):
        for j in range(1, distance_rn.shape[0]):
            dist_rn = distance_rn[i, j] / max_rn
            dist_r2 = distance_r2[i, j] / max_r2

            num = num + (dist_rn - dist_r2) * (dist_rn - dist_r2)
            den = den + dist_rn * dist_rn

    result = num / den

*/

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
        for(y=0;y<instances;y++){
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
        for(y=0;y<instances;y++){
            dist_rn = distance_rn[x][y] / max_rn;
            dist_r2 = distance_r2[x][y] / max_r2;

            num = num + (dist_rn - dist_r2) * (dist_rn - dist_r2);
            den = den + dist_rn * dist_rn;
        }
    }

    printf("Stress: %lf\n", num / den);
}