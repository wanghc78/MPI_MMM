/*
 * utility.c
 *
 *  Created on: Nov 12, 2013
 *      Author: Administrator
 */

#include<stdio.h>
#include<stdlib.h>
#include <mpi.h>
#include "utility.h"


int get_problem_size(int argc, char* argv[], int p, int me) {
    int n = p; //problem size
    if(argc > 1) {
      n = atoi(argv[1]);
      /*Padding the matrix to so that each process get the same size*/
      int per_n = (n - 1) / p + 1;
      if(per_n * p != n) {
          if(me == 0) {
              fprintf(stdout, "[MMM1DRow]Warning: Padding the problem size from %d to %d, Grid size is %d!\n", n, per_n * p, p);
          }
          n = per_n * p;
      } else {
          if(me == 0) {
              fprintf(stdout, "[MMM1DRow]Problem size is %d, Grid size is %d!\n", n, p);
          }
      }
    }
    return n;
}


/*
 * Construct a sub array type for scatter and gather data
 */
void init_subarrtype(int root, int me,
        int n, int dim_sz, int per_n,
        MPI_Datatype* subarrtype_addr, int sendcounts[], int displs[]) {
    int sizes[2]    = {n, n};         /* global size */
    int subsizes[2] = {per_n, per_n}; /* local size */
    int starts[2]   = {0,0};          /* where this one starts */

    MPI_Datatype type;
    mpi_check(MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type));
    mpi_check(MPI_Type_create_resized(type, 0, per_n*sizeof(double), subarrtype_addr));
    mpi_check(MPI_Type_commit(subarrtype_addr));
    int i,j;
    if(me == root) {
        for (i=0; i< dim_sz*dim_sz; i++) { sendcounts[i] = 1; }
        int disp = 0;
        for (i=0; i<dim_sz; i++) {
            for (j=0; j<dim_sz; j++) {
                displs[i*dim_sz+j] = disp;
                disp += 1;
            }
            disp += (per_n-1)*dim_sz;
        }
    }
}


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
            C_ref[i*n+j] = 0;
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
    if(!transposed) {
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
