/*
 * utility.c
 *
 *  Created on: Nov 12, 2013
 *      Author: Administrator
 */

#include<stdio.h>
#include<stdlib.h>
#include "utility.h"

void initial_matrix(double** A_addr, double** BT_addr, double** C_addr, int m, int n, int c) {
    double* A = (double*)malloc(sizeof(double) * m * c);
    double* BT = (double*)malloc(sizeof(double) * n * c);
    double* C = (double*)malloc(sizeof(double) * m * n);
    *A_addr = A;
    *BT_addr = BT;
    *C_addr = C;
    //use random number for input
    int i,j,k;

    srand((unsigned)time(NULL));
    for(i = 0; i < m; i++) {
        for(k = 0; k < c; k++) {
            A[i*c+k] = ((double)rand()/(double)RAND_MAX);
        }
    }
    for(j = 0; j < n; j++) {
        for(k = 0; k < c; k++) {
            BT[j*c+k] = ((double)rand()/(double)RAND_MAX);
        }
    }
}



int check_result(double* A, double* BT, double* C, int m, int n, int c, int transposed) {
    int err_c = 0; //how many errors found
    int i, j, k;
    double* C_ref = (double*)calloc(m * n, sizeof(double)); //with zeroed

    for(i = 0; i < m; i ++) {
        for(j = 0; j < n; j++) {
            for(k = 0; k < c; k++) {
                C_ref[i*n+j] += A[i*c+k] * BT[j*c+k];
            }
        }
    }

    if(transposed){
        for(i = 0; i < m; i++) {
            for(j = 0; j < n; j++) {
                err_c += (fabs(C[j*m + i] - C_ref[i*n+j]) < 0.0001 ? 0 : 1);
            }
        }
    } else {
        //do compare
        for(i = 0; i < m * n ; i++) {
            err_c += (fabs(C[i] - C_ref[i]) < 0.0001 ? 0 : 1);
        }
    }
    free(C_ref);
    return err_c;
}


void print_matrix(char* name, double* A, int m, int n, int transposed) {
    int i,j;
    printf("Matrix %s[%d][%d]\n", name, m, n);
    if(transposed) {
        for(i = 0; i < m; i++) {
            for( j = 0; j < n; j++) {
                printf("%8.2f, ", A[i*n+j]); //note the R is column first
            }
            printf("\n");
        }
    } else {
        for(i = 0; i < m; i++) {
            for( j = 0; j < n; j++) {
                printf("%8.2f, ", A[i+j*m]); //note the R is column first
            }
            printf("\n");
        }
    }

}

void free_matrix(double* A, double* B, double *C) {
    free(A);
    free(B);
    free(C);
}
