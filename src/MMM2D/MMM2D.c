/*
 ============================================================================
 Name        : MMM2D.c
 Author      : Haichuan Wang
 Version     :
 Copyright   : Your copyright notice
 Description : Hello MPI World in C 
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "utility.h"
#include "mkl.h"

void scatter_data(int n, int n_local, int coords[2], int root,
        MPI_Comm cartcomm, MPI_Comm rowcomm, MPI_Comm colcomm,
        double A[n_local*n], double BT[n_local*n],
        double* M, double* NT) {
    //just two steps, first scatter, then
    if (coords[1] == 0) { //join then scatter from the column side, M->A
        mpi_check(
                MPI_Scatter(M, n_local*n, MPI_DOUBLE, A, n_local*n, MPI_DOUBLE, root, colcomm));
    }
    if (coords[0] == 0) { //join the scatter from the row side, BT
        mpi_check(
                MPI_Scatter(NT, n_local*n, MPI_DOUBLE, BT, n_local*n, MPI_DOUBLE, root, rowcomm));
    }
}

void gather_result(int root, int me, int n, int p_gridsize, int n_local,
        MPI_Comm cartcomm, int sendcounts[p_gridsize*p_gridsize],  int displs[p_gridsize*p_gridsize], MPI_Datatype subarrtype,
        double C[n_local*n_local],
        double* M, double* NT, double* P) {

    mpi_check(MPI_Gatherv(C, n_local*n_local,  MPI_DOUBLE,
                 P, sendcounts, displs, subarrtype,
                 root, cartcomm));

    //all in root now
    if(me == root) {
        int err_c = check_result(M, NT, P, n, n, n, 0); //not transposed
        if (err_c) {
            fprintf(stderr, "[MMM2D]Check Failure: %d errors!!!\n", err_c);
            //print_matrix("P", P, n, n, 0);
        } else {
            printf("[MMM2D]Result is verified!\n");
            //print_matrix("C", P, n, n, 0);
        }
        free_matrix(M, NT, P);
    }

}

int main(int argc, char* argv[]) {
    int me; /* rank of process */
    int p; /* number of processes */
    int root = 0; /* rank of sender */
    int tag = 0; /* tag for messages */
    MPI_Status status; /* return status for receive */

    /* start up MPI */
    MPI_Init(&argc, &argv);

    /* find out process rank */
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &me));
    /* find out number of processes */
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &p));

    /*p must be some number's square */
    int p_gridsize = (int)sqrt(p);
    if( p_gridsize * p_gridsize != p) {
      if (me == root){
          fprintf(stderr, "Process Number %d is not a square number!!!\n", p);
      }
      MPI_Finalize();
      exit(1);
    }

    int n = get_problem_size(argc, argv, p_gridsize, me);
    int n_local = n / p_gridsize;

    //prepare for the cartesian topology
    MPI_Comm cartcomm;
    int nbrs[4], dims[2] = {p_gridsize, p_gridsize};
    int periods[2] = {0,0}, reorder=0,coords[2];
    mpi_check(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm));
    MPI_Comm_rank(cartcomm, &me);
    MPI_Cart_coords(cartcomm, me, 2, coords);


    //Split the cartcomm into rowcomm, colcomm
    MPI_Comm rowcomm, colcomm;
    MPI_Comm_split(cartcomm, coords[0], coords[1], &rowcomm);
    MPI_Comm_split(cartcomm, coords[1], coords[0], &colcomm);

    double *M, *NT, *P;
    int root_coords[2] = {0,0}; //with zero first
    MPI_Cart_rank(cartcomm, root_coords, &root); //get the dst
    if (root == me) {
        initial_matrix(&M, &NT, &P, n, n, n);
    }

    //now each process has
    /*The matrix size is p * p*/
    int i, j, k;
    double *A = (double*)mkl_malloc(n_local * n*sizeof(double), 16); //each has n_local row, i=me, k 0:(p-1)
    double *BT = (double*)mkl_malloc(n_local * n*sizeof(double), 16); //j, k
    double *C = (double*)mkl_malloc(n_local * n_local * sizeof(double), 16);

    //just two steps, first scatter, then
    scatter_data(n, n_local, coords, root,
            cartcomm,  rowcomm, colcomm,
            A, BT, M, NT);

    mpi_check(MPI_Barrier(cartcomm));
    double t0, t1;
    //Start timing point
    t0 = MPI_Wtime();
    //The real start point.

    //now do A's broadcast and B's broadcat
    mpi_check(MPI_Bcast(A, n_local*n, MPI_DOUBLE, 0, rowcomm));
    mpi_check(MPI_Bcast(BT, n_local*n, MPI_DOUBLE, 0, colcomm));


    //now do the vector vector calculation
//    for(i = 0; i < n_local; i++) {
//        for(j = 0; j < n_local; j++) {
//            C[i*n_local+j] = 0;
//            for(k = 0; k < n; k++) {
//                C[i*n_local+j] += A[i*n+k] * BT[j*n+k];
//            }
//        }
//    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            n_local, n_local, n,
            1, A, n, BT, n, 0, C, n_local);

    //End timing
    t1 = MPI_Wtime() - t0;
    //end timing point
    //use reduction to collect the final time
    MPI_Reduce(&t1, &t1, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if(me == root) {
        printf("[mmm2D]P=%d, N=%d, Time=%.9f\n", p, n, t1/p);
    }


    //Scatter and Gather used data types
    MPI_Datatype subarrtype;
    int sendcounts[p_gridsize*p_gridsize];
    int displs[p_gridsize*p_gridsize]; //value in block (resized) count
    init_subarrtype(root, me, n, p_gridsize, n_local, &subarrtype, sendcounts, displs);
#ifdef VERIFY
    gather_result(root, me, n, p_gridsize, n_local,
            cartcomm, sendcounts, displs, subarrtype,
            C, M, NT, P);
#endif

    mkl_free(A); mkl_free(BT); mkl_free(C);
    MPI_Finalize();
    return 0;
}
