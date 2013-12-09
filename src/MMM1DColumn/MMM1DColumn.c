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
#include "mkl.h"

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
    int n_local = n / p;

    /*The matrix size is n * n*/
    int i, j, k;
    double* A = mkl_malloc(n * n * sizeof(double), 16);
    double* BT = mkl_malloc(n_local * n * sizeof(double), 16);
    double* C = mkl_malloc(n * n_local * sizeof(double), 16);

    double* M, *NT, *P;
    if (me == root) {
        initial_matrix(&M, &NT, &P, n, n, n);
        //initialize A
        memcpy(A, M, sizeof(double) * n * n); //copy M->A
    }

    //use scatter to initial all BT
    mpi_check(MPI_Scatter(NT, n_local * n, MPI_DOUBLE, BT, n_local * n, MPI_DOUBLE, root, MPI_COMM_WORLD));
    //now BT is ready

    double t0, t1;
    //Start timing point
    t0 = MPI_Wtime();

    //Broadcast from p == 0 to all others
    mpi_check(MPI_Bcast(A, n*n, MPI_DOUBLE, root, MPI_COMM_WORLD));
    //now all has A ready

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
            n, n_local, n,
            1, A, n, BT, n, 0, C, n_local);

    t1 = MPI_Wtime() - t0;
    //end timing point
    //use reduction to collect the final time
    MPI_Reduce(&t1, &t1, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if(me == root) {
        printf("[mmm1DColumn]P=%d, N=%d, Time=%.9f\n", p, n, t1/p);
    }

    //Final collection of all the result to rank 0.
    //In order to get the right gather, I use MPI_Type_vector
    MPI_Datatype colrawtype, coltype;
    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &colrawtype);
    MPI_Type_commit(&colrawtype);
    MPI_Type_create_resized(colrawtype, 0, 1*sizeof(double), &coltype);
    MPI_Type_commit(&coltype);

    // Note local_* type use n*n_local size matrix
    MPI_Datatype local_colrawtype, local_coltype;
    MPI_Type_vector(n, 1, n_local, MPI_DOUBLE, &local_colrawtype);
    MPI_Type_commit(&local_colrawtype);
    MPI_Type_create_resized(local_colrawtype, 0, 1*sizeof(double), &local_coltype);
    MPI_Type_commit(&local_coltype);

    //Just store the result
    mpi_check(MPI_Gather(C, n_local, local_coltype, P, n_local, coltype, root, MPI_COMM_WORLD));

#ifdef VERIFY
    if(me == root) {
        int err_c = check_result(M, NT, P, n, n, n, 0);
        if(err_c) {
            fprintf(stderr, "[MMM1DColumn]Check Failure: %d errors!!!\n", err_c);
            print_matrix("A", M, n, n, 1);
            print_matrix("B", NT, n, n, 1);
            print_matrix("C", P, n, n, 1);
        } else {
            printf("[MMM1DColumn]Result is verified!\n");
            //print_matrix("C", P, n, n, 0);
        }
        free_matrix(M, NT, P);
    }
#endif
    mkl_free(A);mkl_free(BT); mkl_free(C);
    MPI_Finalize();
    return 0;
}
