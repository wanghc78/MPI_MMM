/*
 ============================================================================
 Name        : MMM3D.c
 Author      : Haichuan Wang
 Version     :
 Copyright   : Your copyright notice
 Description : Matrix Matrix M
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "utility.h"
#include "mkl.h"


void scatter_data(int root, int me, int coords[3],
        int n, int p_gridsize, int n_local,
        MPI_Comm cartcomm, int sendcounts[p_gridsize*p_gridsize],  int displs[p_gridsize*p_gridsize], MPI_Datatype subarrtype,
        double A[n_local * n_local], double BT[n_local * n_local],
        double* M, double* NT) {

    MPI_Comm dim_ik_comm, dim_jk_comm;
    mpi_check(MPI_Comm_split(cartcomm, coords[1], coords[0]*p_gridsize+coords[2], &dim_ik_comm)); //ik as rank
    mpi_check(MPI_Comm_split(cartcomm, coords[0], coords[1]*p_gridsize+coords[2], &dim_jk_comm)); //jk as rank

    if(coords[1] == 0) {
        mpi_check(MPI_Scatterv(M, sendcounts,  displs, subarrtype,
                A, n_local*n_local, MPI_DOUBLE, 0, dim_ik_comm));
    }

    if(coords[0] == 0) {
        mpi_check(MPI_Scatterv(NT, sendcounts,  displs, subarrtype,
                BT, n_local*n_local, MPI_DOUBLE, 0, dim_jk_comm));
    }

    MPI_Comm_free(&dim_ik_comm);
    MPI_Comm_free(&dim_jk_comm);

}

void gather_result(int root, int me, int coords[3],
        int n, int p_gridsize, int n_local,
        MPI_Comm cartcomm, MPI_Comm dim_ij_comm, int sendcounts[p_gridsize*p_gridsize],  int displs[p_gridsize*p_gridsize], MPI_Datatype subarrtype,
        double C[n_local*n_local],
        double* M, double* NT, double* P) {


    if (coords[2] == 0) { //only the k==0 join the merge
        mpi_check(MPI_Gatherv(C, n_local*n_local,  MPI_DOUBLE,
                     P, sendcounts, displs, subarrtype,
                     root, dim_ij_comm));
    }

    if(me == root) {
        int err_c = check_result(M, NT, P, n, n, n, 0); //not transposed
        if (err_c) {
            fprintf(stderr, "[MMM3D]Check Failure: %d errors!!!\n", err_c);
            //print_matrix("P", P, n, n, 0);
        } else {
            printf("[MMM3D]Result is verified!\n");
            //print_matrix("C", P, n, n, 0);
        }
        free_matrix(M, NT, P);
    }
}




int main(int argc, char* argv[]) {
    int me; /* rank of process */
    int p; /* number of processes */
    int src; /* rank of sender */
    int root = 0;
    int tag = 0; /* tag for messages */
    MPI_Status status; /* return status for receive */

    /* start up MPI */
    MPI_Init(&argc, &argv);

    /* find out process rank */
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &me));
    /* find out number of processes */
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &p));

    /*p must be some number's square */
    int p_gridsize = (int)pow(p,1.0/3.0);
    if( p_gridsize * p_gridsize * p_gridsize != p) {
      if (me == 0 ){
          fprintf(stderr, "Process Number %d is not a cubic number!!!\n", p);
      }
      MPI_Finalize();
      exit(1);
    }

    int n = get_problem_size(argc, argv, p_gridsize, me);
    int n_local = n / p_gridsize;

    //prepare for the cartesian topology
    MPI_Comm cartcomm;
    int nbrs[6], dims[3] = {p_gridsize, p_gridsize, p_gridsize};
    int periods[3] = {0,0,0}, reorder=0, coords[3];
    mpi_check(MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cartcomm));
    MPI_Comm_rank(cartcomm, &me);
    MPI_Cart_coords(cartcomm, me, 3, coords);

    //Split the cartcomm into rowcomm(hor-broadcast), colcomm(vertical broadcast), depthcomm (z dim reduction)
    MPI_Comm rowcomm, colcomm, depthcomm;
    MPI_Comm_split(cartcomm, coords[0]*p_gridsize+coords[2], coords[1], &rowcomm); //j as rank
    MPI_Comm_split(cartcomm, coords[1]*p_gridsize+coords[2], coords[0], &colcomm); //i as rank
    MPI_Comm_split(cartcomm, coords[0]*p_gridsize+coords[1], coords[2], &depthcomm); //k as rank

    double *M, *NT, *P;
    int root_coords[3]={0,0,0}; //The src ==0's node should prepare the data
    MPI_Cart_rank(cartcomm, root_coords, &root); //get the dst
    if(root == me) { //I'm the {0,0,0} node
        initial_matrix(&M, &NT, &P, n, n, n);
    }

    //now each process has
    /*The matrix size is p * p*/
    int i, j, k;
    double *A = (double*)mkl_malloc(n_local * n_local * sizeof(double), 16); //each on has 1
    double *BT = (double*)mkl_malloc(n_local * n_local * sizeof(double), 16); //each only has 1
    double *C = (double*)mkl_malloc(n_local * n_local * sizeof(double), 16); //only n_local * n_local

    //Scatter and Gather used data types
    MPI_Datatype subarrtype;
    int sendcounts[p_gridsize*p_gridsize];
    int displs[p_gridsize*p_gridsize]; //value in block (resized) count
    init_subarrtype(root, me, n, p_gridsize, n_local, &subarrtype, sendcounts, displs);


    //Send data from root {0, 0, 0} to A{j=0 array}, BT{i=0 array}
    scatter_data(root, me, coords, n, p_gridsize, n_local,
            cartcomm, sendcounts, displs, subarrtype,
            A, BT, M, NT);

    mpi_check(MPI_Barrier(cartcomm));
    double t0, t1;
    //Start timing point
    t0 = MPI_Wtime();
    //do broadcast for A/B
    mpi_check(MPI_Bcast(A, n_local*n_local, MPI_DOUBLE, 0, rowcomm));
    mpi_check(MPI_Bcast(BT, n_local*n_local, MPI_DOUBLE, 0, colcomm));


    //do calculation
//    for(i = 0; i < n_local; i++) {
//        for(j = 0; j < n_local; j++) {
//            C[i*n_local+j] = 0;
//            for(k = 0; k < n_local; k++) {
//                C[i*n_local+j] += A[i*n_local+k] * BT[j*n_local+k];
//            }
//        }
//    }

    //use mkl
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            n_local, n_local, n_local,
            1, A, n_local, BT, n_local, 0, C, n_local);


    // Way 2 - Use MPI_Reduce

    mpi_check(MPI_Reduce(C, C, n_local*n_local, MPI_DOUBLE, MPI_SUM, 0, depthcomm));

    //End timing
    t1 = MPI_Wtime() - t0;
    //node the end time should be in depthcomm with k == 0
    //end timing point
    //use reduction to collect the final time
    MPI_Comm dim_ij_comm;
    MPI_Comm_split(cartcomm, coords[2], coords[0]*p_gridsize+coords[1], &dim_ij_comm); //ij as rank


    if(coords[2] == 0) {
        MPI_Reduce(&t1, &t1, 1, MPI_DOUBLE, MPI_SUM, root, dim_ij_comm);
        if(me == root) {
            printf("[mmm3D]P=%d, N=%d, Time=%.9f\n", p, n, t1/(p_gridsize*p_gridsize));
        }
    }
#ifdef VERIFY
    gather_result(root, me, coords, n, p_gridsize, n_local,
            cartcomm, dim_ij_comm, sendcounts, displs, subarrtype,
            C, M, NT, P);
#endif
    MPI_Comm_free(&dim_ij_comm);
    mkl_free(A); mkl_free(BT); mkl_free(C);
    MPI_Type_free(&subarrtype);
    mpi_check(MPI_Finalize());
    return 0;
}
