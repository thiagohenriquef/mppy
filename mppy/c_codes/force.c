#include <stdlib.h>
#include <math.h>
#include <stdio.h>

int * permutation(int size){
    int *elements = malloc(sizeof(int)*size);

    // inizialize
    for (int i = 0; i < size; ++i)
      elements[i] = i;

    for (int i = size - 1; i > 0; --i) {
      // generate random index
      int w = rand()%i;
      // swap items
      int t = elements[i];
      elements[i] = elements[w];
      elements[w] = t;
    }

    return elements;
}
// create a 2d array from the 1d one
float ** convert2d(int len1, int len2, float * arr) {
    float ** ret_arr;

    // allocate the additional memory for the additional pointers
    ret_arr = (float **)malloc(sizeof(float*)*len1);

    // set the pointers to the correct address within the array
    for (int i = 0; i < len1; i++) {
        ret_arr[i] = &arr[i*len2];
    }

    // return the 2d-array
    return ret_arr;
}

// print the 2d array
void print_2d_list(int len1,
    int len2,
    float * list) {

    // call the 1d-to-2d-conversion function
    float ** list2d = convert2d(len1, len2, list);

    // print the array just to show it works
    for (int index1 = 0; index1 < len1; index1++) {
        for (int index2 = 0; index2 < len2; index2++) {
            printf("%1.1f ", list2d[index1][index2]);
        }
        printf("\n");
    }

    // free the pointers (only)
    free(list2d);
}


extern void force (double **dist,
                double **Y,
                int instances,
                int max_iter,
                double eps,
                double delta_frac)
                {
    int i;
    int j;
    int k;
    int instance1, instance2;
    double dr2 = 0.0, drn = 0.0, delta = 0.0;
    double x1x2 = 0.0, y1y2=0.0;
    int *index;
    index = permutation(instances);

    for (i=0; i < max_iter; i++){
        for (j=0; j < instances; j++){
            instance1 = index[j];
            
            for (k=0; k < instances; k++){
                instance2 = index[k];
                
                if (instance1 == instance2){
                    continue;
                }else{
                    x1x2 = Y[instance2][0] - Y[instance1][0];
                    //printf("%f\n", x1x2);
                    y1y2 = Y[instance2][1] - Y[instance1][1];
                    dr2 = sqrt((x1x2 * x1x2) + (y1y2 * y1y2));
                    
                }

                if(dr2 < eps){
                    dr2 = eps;
                }

                drn = dist[instance1][instance2] - dr2;
                delta = drn - dr2;
                delta /= delta_frac;

                Y[instance2][0] += delta * (x1x2 / dr2);
                Y[instance2][1] += delta * (y1y2 / dr2);
            }
        }
    }
}