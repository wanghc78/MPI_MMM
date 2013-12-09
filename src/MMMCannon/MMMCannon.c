/*
 ============================================================================
 Name        : MMMCannon.c
 Author      : Haichuan Wang
 Version     :
 Copyright   : Your copyright notice
 Description : Cannon Algorithm MMM2D
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "utility.h"
#include "mkl.h"


void scatter_data(int root, int me,
        int n, int p_gridsize, int n_local,
        MPI_Comm cartcomm, int sendcounts[p_gridsize*p_gridsize],  int displs[p_gridsize*p_gridsize], MPI_Datatype subarrtype,
        double A[n_local*n_local], double BT[n_local*n_local],
        double* M, double* NT) {
    mpi_check(MPI_Scatterv(M, sendcounts,  displs, subarrtype,
            A, n_local*n_local, MPI_DOUBLE, root, cartcomm));

    //a small trick to transpose displs -> displsBT
    int displsBT[p_gridsize*p_gridsize];
    int i,j;
    if(root == me) {
        for(i = 0; i < p_gridsize; i++) {
            for(j = 0; j < p_gridsize; j++) {
                displsBT[i*p_gridsize+j] = displs[j*p_gridsize+i];
            }
        }
    }


    mpi_check(MPI_Scatterv(NT, sendcounts,  displsBT, subarrtype,
            BT, n_local*n_local, MPI_DOUBLE, root, cartcomm));

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
            fprintf(stderr, "[MMM2D]Check Failure: %d errors\n", err_c);
            //print_matrix("P", P, n, n, 0);
        } else {
            printf("[MMM2D]Result is verified\n");
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
          fprintf(stderr, "Process Number %d is not a square number\n", p);
      }
      MPI_Finalize();
      exit(1);
    }

    int n = get_problem_size(argc, argv, p_gridsize, me);
    int n_local = n / p_gridsize;

    //prepare for the cartesian topology
    MPI_Comm cartcomm;
    int nbrs[4], dims[2] = {p_gridsize, p_gridsize};
    int periods[2] = {1,1}, reorder=0,coords[2];
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

    //now each process has a small portion of local
    /*The matrix size is p * p*/
    int i, j, k;
    double *A = (double*)mkl_malloc(n_local * n_local * sizeof(double), 16);
    double *BT = (double*)mkl_malloc(n_local * n_local * sizeof(double), 16); //j, k
    double *C = (double*)mkl_malloc(n_local * n_local * sizeof(double), 16);

    //Scatter and Gather used data types
    MPI_Datatype subarrtype;
    int sendcounts[p_gridsize*p_gridsize];
    int displs[p_gridsize*p_gridsize]; //value in block (resized) count
    init_subarrtype(root, me, n, p_gridsize, n_local, &subarrtype, sendcounts, displs);

    //Now A/BT local are ready
    scatter_data(root, me, n, p_gridsize, n_local,
            cartcomm, sendcounts, displs, subarrtype,
            A, BT, M, NT);

    mpi_check(MPI_Barrier(cartcomm));
    double t0, t1;
    //Start timing point
    t0 = MPI_Wtime();
    //The real start point.

    //the main Cannon algorithm part

    //first need shift both A and B.
    int dst, src; //use for shift
    int datatag = 0;
    //For the i-th row of the subtasks grid the matrix  A blocks are shifted for (i-1) positions to the left,
    int shift = coords[0];
    if(shift > 0) {
        dst = (coords[1] +  p_gridsize - shift) % p_gridsize;
        src = (coords[1] + shift) % p_gridsize;
        MPI_Sendrecv_replace(A, n_local*n_local, MPI_DOUBLE, dst,
        datatag, src, datatag, rowcomm, &status);
    }

    shift = coords[1];
    if(shift > 0) {
        dst = (coords[0] +  p_gridsize - shift) % p_gridsize;
        src = (coords[0] + shift) % p_gridsize;
        MPI_Sendrecv_replace(BT, n_local*n_local, MPI_DOUBLE, dst,
        datatag, src, datatag, colcomm, &status);
    }



    int j_dst = (coords[1] +p_gridsize - 1) % p_gridsize; //used for BT's shift
    int j_src = (coords[1] + 1) % p_gridsize; //used for BT's shift
    int i_dst = (coords[0] +p_gridsize - 1) % p_gridsize; //used for BT's shift
    int i_src = (coords[0] + 1) % p_gridsize; //used for BT's shift
    //reference: http://www.hpcc.unn.ru/mskurs/ENG/PPT/pp08.pdf
    int stage = 0;

//    for(i = 0; i < n_local; i++) {
//        for(j = 0; j < n_local; j++) {
//            C[i*n_local+j] = 0;
//            for(k = 0; k < n_local; k++) {
//                C[i*n_local+j] += A[i*n_local+k] * BT[j*n_local+k];
//            }
//        }
//    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            n_local, n_local, n_local,
            1, A, n_local, BT, n_local, 0, C, n_local);

    for(stage++; stage < p_gridsize; stage++) {

        /* Send submatrix of A left and receive a new from right */
        MPI_Sendrecv_replace(A, n_local*n_local, MPI_DOUBLE, j_dst,
        datatag, j_src, datatag, rowcomm, &status);

        /* Send submatrix of B up and receive a new from below */
        MPI_Sendrecv_replace(BT, n_local*n_local, MPI_DOUBLE, i_dst,
        datatag, i_src, datatag, colcomm, &status);

//        for(i = 0; i < n_local; i++) {
//            for(j = 0; j < n_local; j++) {
//                for(k = 0; k < n_local; k++) {
//                    C[i*n_local+j] += A[i*n_local+k] * BT[j*n_local+k];
//                }
//            }
//        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_local, n_local, n_local,
                1, A, n_local, BT, n_local, 1, C, n_local);
    }

    //End timing
    t1 = MPI_Wtime() - t0;
    //end timing point
    //use reduction to collect the final time
    MPI_Reduce(&t1, &t1, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if(me == root) {
        printf("[mmm2D]P=%d, N=%d, Time=%.9f\n", p, n, t1/p);
    }

#ifdef VERIFY
    gather_result(root, me, n, p_gridsize, n_local,
            cartcomm, sendcounts, displs, subarrtype,
            C, M, NT, P);
#endif
    MPI_Type_free(&subarrtype);
    mkl_free(A); mkl_free(BT); mkl_free(C);
    MPI_Finalize();
    return 0;
}
