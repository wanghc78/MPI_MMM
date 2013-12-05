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
#include "mpi.h"
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
    int i, j, k;
    double A[p]; //each has one row, i=me, k 0:(p-1)
    double BT[p * p]; //k,j
    double C[p * 1]; //i=me, j 0:(p-1)

    double* M, *NT, *P;

    if (me == 0) {
        initial_matrix(&M, &NT, &P, p, p, p);
        memcpy(BT, NT, sizeof(double) * p * p); //copy M->A
    }

    //use scatter to initial all A
    mpi_check(MPI_Scatter(M, p, MPI_DOUBLE, A, p, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    //now A is ready

    //Broadcast from p == 0 to all others
    mpi_check(MPI_Bcast(BT, p*p, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    //now all has BT ready

    //now do the matrix calculation, Note BT is transposed, we juse j,k to iterator
    for(j = 0; j < p; j++) {
        C[j] = 0; //initial
        for(k = 0; k < p; k++) {
            C[j] += A[k] * BT[j*p+k];   //C[i,j],j is p
        }
    }

    //Final collection of all the result to rank 0.
    //Just store the result
    mpi_check(MPI_Gather(C, p, MPI_DOUBLE, P, p, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    if(me == 0) {
        int err_c = check_result(M, NT, P, p, p, p, 0); //not transposed
        if(err_c) {
            fprintf(stderr, "[MMM1DRow]Check Failure: %d errors!!!\n", err_c);
        } else {
            printf("[MMM1DRow]Result is verified!\n");
            //print_matrix("C", P, p, p, 0);
        }
    }
    MPI_Finalize();
    return 0;
}
