/* File:  
 *    main.c
 *
 * Purpose:
 *    Illustrate a multithreaded linear equation solver using the Gauss-Jordan elimination method
 *
 * Input:
 *    Number of threads
 *
 * Usage:    ./main <snumber_threads>
 *
 */
 #include <stdio.h>
#include <stdlib.h>
#include <pthread.h> 
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <omp.h> 
#include "Lab3IO.h"
#include "timer.h"

int thread_count;
int serial(double** Au, int size);
int parallel(double** Au, int size);

/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {

    if (argc != 2){ 
        fprintf(stderr, "usage: %s <number_threads>\n", argv[0]);
        exit(0);
    }

    /* Get number of threads from command line */
    thread_count = atoi(argv[1]);

    /* Load input created by datagen */
    double **Au; int size; 
    Lab3LoadInput(&Au, &size);

    parallel(Au, size);
    //serial(Au, size);
    return 0;
}  /* main */

int parallel(double** Au, int size){
/*Calculate the solution by parallel code*/
    double start, finish;
    GET_TIME(start);
    int i, j, k, ind, l_ind;
	double* X;
	double temp, max, l_max;
	int* index;
	X = CreateVec(size);
    index = malloc(size * sizeof(int));

    for (i = 0; i < size; ++i)
        index[i] = i;

    if (size == 1)
        X[0] = Au[0][1] / Au[0][0];
    else{
        /*Gaussian elimination*/
        # pragma omp parallel num_threads(thread_count) shared(Au, size, X, index, max, ind) private(j, i, temp, k, l_max, l_ind)
        {
            for (k = 0; k < size - 1; ++k){
                /*Pivoting*/
                max = 0;
                ind = 0;
                # pragma omp for
                for (i = k; i < size; i++){ //Find local max and index of max for each thread
                    if (max < Au[index[i]][k] * Au[index[i]][k]){
                        l_max = Au[index[i]][k] * Au[index[i]][k]; 
                        l_ind = i;
                    } 
                }

                if (l_max > max){
                    #pragma omp critical
                    {
                        if (l_max > max){ //Double check still true once in critical
                            max = l_max;
                            ind = l_ind;
                        }
                    }
                }
                # pragma omp barrier
                # pragma omp single
                {
                    if (ind != k)/*swap*/{
                        i = index[ind];
                        index[ind] = index[k];
                        index[k] = i;
                    }
                }
            
                /*Elimination*/
                # pragma omp for
                for (i = k + 1; i < size; ++i){
                    temp = Au[index[i]][k] / Au[index[k]][k];
                    for (j = k; j < size + 1; ++j)
                        Au[index[i]][j] -= Au[index[k]][j] * temp;
                }
            }

            /*Jordan elimination*/
            for (k = size - 1; k > 0; --k){
            
                # pragma omp for //Need fixed k for paralle elimination
                for (i = k - 1; i >= 0; --i ){
                    temp = Au[index[i]][k] / Au[index[k]][k];
                    Au[index[i]][k] -= temp * Au[index[k]][k];
                    Au[index[i]][size] -= temp * Au[index[k]][size];
                }
            }

            /*solution*/
            # pragma omp for
            for (k=0; k< size; ++k)
                X[k] = Au[index[k]][size] / Au[index[k]][k];

        }   
    }
    GET_TIME(finish);
    Lab3SaveOutput(X, size, finish-start);
	
    DestroyVec(X);
    DestroyMat(Au, size);
    free(index);
	return 0;
}

//TO DO remove later once paralled in working
int serial(double** Au, int size){
/*Calculate the solution by serial code*/
    double start, finish;
    GET_TIME(start);
    int i, j, k;
	double* X;
	double temp;
	int* index;


	X = CreateVec(size);
    index = malloc(size * sizeof(int));
    for (i = 0; i < size; ++i)
        index[i] = i;

    if (size == 1)
        X[0] = Au[0][1] / Au[0][0];
    else{
        /*Gaussian elimination*/
        for (k = 0; k < size - 1; ++k){
            /*Pivoting*/
            temp = 0;
            for (i = k, j = 0; i < size; ++i)
                if (temp < Au[index[i]][k] * Au[index[i]][k]){
                    temp = Au[index[i]][k] * Au[index[i]][k];
                    j = i;
                }
            if (j != k)/*swap*/{
                i = index[j];
                index[j] = index[k];
                index[k] = i;
            }
            /*Elimination*/
            for (i = k + 1; i < size; ++i){
                temp = Au[index[i]][k] / Au[index[k]][k];
                for (j = k; j < size + 1; ++j)
                    Au[index[i]][j] -= Au[index[k]][j] * temp;
            }       
        }
        /*Jordan elimination*/
        for (k = size - 1; k > 0; --k){
            for (i = k - 1; i >= 0; --i ){
                temp = Au[index[i]][k] / Au[index[k]][k];
                Au[index[i]][k] -= temp * Au[index[k]][k];
                Au[index[i]][size] -= temp * Au[index[k]][size];
            } 
        }
        /*solution*/
        for (k=0; k< size; ++k)
            X[k] = Au[index[k]][size] / Au[index[k]][k];
    }

    GET_TIME(finish);
    Lab3SaveOutput(X, size, finish-start);

	
    DestroyVec(X);
    DestroyMat(Au, size);
    free(index);
	return 0;
}
