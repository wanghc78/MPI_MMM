/*
 ============================================================================
 Name        : MMM1DColumn.c
 Author      : Haichuan Wang
 Version     :
 Copyright   : Your copyright notice
 Description : Hello MPI World in C 
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "utility.h"

int main(int argc, char* argv[]) {
    int me; /* rank of process */
    int p; /* number of processes */
    int source; /* rank of sender */
    int dest; /* rank of receiver */
    int tag = 0; /* tag for messages */
    MPI_Status status; /* return status for receive */

    /* start up MPI */
    MPI_Init(&argc, &argv);

    /* find out process rank */
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &me));
    /* find out number of processes */
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &p));

    /*The matrix size is p * p*/
    //Each p has full A[p*p] i,k
    //Each p has a portional of B[p*1], k,j
    //Result stored at each p by C[p*1] i,j
    int i, j, k;
    double A[p * p];
    double B[p * 1];
    double C[p * 1];

    double* M, *NT, *P;
    if (me == 0) {
        initial_matrix(&M, &NT, &P, p, p, p);
        //initialize A
        memcpy(A, M, sizeof(double) * p * p); //copy M->A
    }
    //Broadcast from p == 0 to all others
    mpi_check(MPI_Bcast(A, p*p, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    //now all has A ready

    //use scatter to initial all B
    mpi_check(MPI_Scatter(NT, p, MPI_DOUBLE, B, p, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    //now B is ready

    //now do the matrix calculation
    for(i = 0; i < p; i++) {
        C[i] = 0; //initial
        for(k = 0; k < p; k++) {
            C[i] += A[i*p+k] * B[k];   //C[i,j],j is p
        }
    }
    //Final collection of all the result to rank 0.
    //Just store the result
    mpi_check(MPI_Gather(C, p, MPI_DOUBLE, P, p, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    if(me == 0) {
        int err_c = check_result(M, NT, P, p, p, p, 1);
        if(err_c) {
            fprintf(stderr, "[MMM1DColumn]Check Failure: %d errors!!!\n", err_c);
        } else {
            printf("[MMM1DColumn]Result is verified!\n");
            //print_matrix("C", P, p, p, 1);
        }

    }
    MPI_Finalize();
    return 0;
}
