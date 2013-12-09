/*
 ============================================================================
 Name        : MMMSerial.c
 Author      : Haichuan Wang
 Version     :
 Copyright   : Your copyright notice
 Description : Hello MPI World in C 
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include "utility.h"
#include "mkl.h"

int main(int argc, char* argv[]) {


    /* Decide the problem size */
    int n = get_problem_size(argc, argv, 1, 0);

    /*The matrix size is n * n*/
    int i, j, k;
    double* A = mkl_malloc(n * n * sizeof(double), 16);
    double* BT = mkl_malloc(n * n * sizeof(double), 16);
    double* C = mkl_malloc(n * n * sizeof(double), 16);
    initial_matrix(&A, &BT, &C, n, n, n);



    double t0, t1;
    //Start timing point
    t0 = dsecnd();


    //now do the matrix calculation
//    for(i = 0; i < n; i++) {
//        for(j = 0; j < n_local; j++){
//            C[i*n_local+j] = 0; //initial
//            for(k = 0; k < n; k++) {
//                C[i*n_local+j] += A[i*n+k] * BT[j*n+k];   //C[i,j],j is p
//            }
//        }
//    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            n, n, n,
            1, A, n, BT, n, 0, C, n);

    t1 = dsecnd() - t0;
    printf("[MMMSerial]P=%d, N=%d, Time=%.9f\n", 1, n, t1);

    mkl_free(A);mkl_free(BT); mkl_free(C);
    return 0;
}
