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

//test
int main(int argc, char* argv[]) {
    int me; /* rank of process */
    int p; /* number of processes */
    int root = 0;
    int tag = 0; /* tag for messages */
    MPI_Status status; /* return status for receive */

    /* start up MPI */
    MPI_Init(&argc, &argv);

    /* find out process rank */
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &me));
    /* find out number of processes */
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &p));

    /* Decide the problem size */
    int n = get_problem_size(argc, argv, p, me);
    int per_n = n / p;

    /*The matrix size is n * n*/
    int i, j, k;
    double* A = malloc(per_n * n * sizeof(double)); //each has per_n row, i=me, k 0:(p-1)
    double* BT = malloc(n * n * sizeof(double)); //k,j
    double* C = malloc(per_n * n* sizeof(double));; //i=me, j 0:(p-1)

    double* M, *NT, *P;

    if (me == root) {
        initial_matrix(&M, &NT, &P, n, n, n);
        memcpy(BT, NT, sizeof(double) * n * n); //copy M->A
    }

    //use scatter to initial all A
    mpi_check(MPI_Scatter(M, n * per_n, MPI_DOUBLE, A, n * per_n, MPI_DOUBLE, root, MPI_COMM_WORLD));
    //now A is ready

    double t0, t1;
    //Start timing point
    t0 = MPI_Wtime();

    //Broadcast from p == 0 to all others
    mpi_check(MPI_Bcast(BT, n*n, MPI_DOUBLE, root, MPI_COMM_WORLD));
    //now all has BT ready

    //now do the matrix calculation, Note BT is transposed, we juse j,k to iterator

    for(i = 0; i < per_n; i++) {
        for(j = 0; j < n; j++) {
            C[i*n + j] = 0; //initial
            for(k = 0; k < n; k++) {
                C[i*n + j] += A[i*n + k] * BT[j*n+k];   //C[i,j],j is p
            }
        }
    }

    t1 = MPI_Wtime() - t0;
    //end timing point
    //use reduction to collect the final time
    MPI_Reduce(&t1, &t1, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if(me == root) {
        printf("[mmm1DRow]P=%d, N=%d, Time=%.9f\n", p, n, t1/p);
    }


    //Final collection of all the result to rank 0.
    //Just store the result
    mpi_check(MPI_Gather(C, per_n * n, MPI_DOUBLE, P, per_n * n, MPI_DOUBLE, root, MPI_COMM_WORLD));
    if(me == root) {
        int err_c = check_result(M, NT, P, n, n, n, 0); //not transposed
        if(err_c) {
            fprintf(stderr, "[MMM1DRow]Check Failure: %d errors!!!\n", err_c);
        } else {
            printf("[MMM1DRow]Result is verified!\n");
            //print_matrix("C", P, n, n, 0);
        }
    }
    free(A);free(BT); free(C);
    MPI_Finalize();
    return 0;
}
