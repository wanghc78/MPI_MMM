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

#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3
#define FRONT 4
#define BACK  5



void scatter_data(int root, int me, int coords[3],
        int n, int dim_sz, int per_n,
        MPI_Comm cartcomm, int sendcounts[dim_sz*dim_sz],  int displs[dim_sz*dim_sz], MPI_Datatype subarrtype,
        double A[per_n * per_n], double BT[per_n * per_n],
        double* M, double* NT) {

    MPI_Comm dim_ik_comm, dim_jk_comm;
    mpi_check(MPI_Comm_split(cartcomm, coords[1], coords[0]*dim_sz+coords[2], &dim_ik_comm)); //ik as rank
    mpi_check(MPI_Comm_split(cartcomm, coords[0], coords[1]*dim_sz+coords[2], &dim_jk_comm)); //jk as rank

    if(coords[1] == 0) {
        mpi_check(MPI_Scatterv(M, sendcounts,  displs, subarrtype,
                A, per_n*per_n, MPI_DOUBLE, 0, dim_ik_comm));
    }

    if(coords[0] == 0) {
        mpi_check(MPI_Scatterv(NT, sendcounts,  displs, subarrtype,
                BT, per_n*per_n, MPI_DOUBLE, 0, dim_jk_comm));
    }

    MPI_Comm_free(&dim_ik_comm);
    MPI_Comm_free(&dim_jk_comm);

}

void gather_result(int root, int me, int coords[3],
        int n, int dim_sz, int per_n,
        MPI_Comm cartcomm, MPI_Comm dim_ij_comm, int sendcounts[dim_sz*dim_sz],  int displs[dim_sz*dim_sz], MPI_Datatype subarrtype,
        double C[per_n*per_n],
        double* M, double* NT, double* P) {


    if (coords[2] == 0) { //only the k==0 join the merge
        mpi_check(MPI_Gatherv(C, per_n*per_n,  MPI_DOUBLE,
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
    int dim_sz = (int)pow(p,1.0/3.0);
    if( dim_sz * dim_sz * dim_sz != p) {
      if (me == 0 ){
          fprintf(stderr, "Process Number %d is not a cubic number!!!\n", p);
      }
      MPI_Finalize();
      exit(1);
    }

    int n = get_problem_size(argc, argv, dim_sz, me);
    int per_n = n / dim_sz;

    //prepare for the cartesian topology
    MPI_Comm cartcomm;
    int nbrs[6], dims[3] = {dim_sz, dim_sz, dim_sz};
    int periods[3] = {0,0,0}, reorder=0, coords[3];
    mpi_check(MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cartcomm));
    MPI_Comm_rank(cartcomm, &me);
    MPI_Cart_coords(cartcomm, me, 3, coords);
    MPI_Cart_shift(cartcomm, 0, 1, &nbrs[UP], &nbrs[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);
    MPI_Cart_shift(cartcomm, 2, 1, &nbrs[FRONT], &nbrs[BACK]);

    //Split the cartcomm into rowcomm(hor-broadcast), colcomm(vertical broadcast), depthcomm (z dim reduction)
    MPI_Comm rowcomm, colcomm, depthcomm;
    MPI_Comm_split(cartcomm, coords[0]*dim_sz+coords[2], coords[1], &rowcomm); //j as rank
    MPI_Comm_split(cartcomm, coords[1]*dim_sz+coords[2], coords[0], &colcomm); //i as rank
    MPI_Comm_split(cartcomm, coords[0]*dim_sz+coords[1], coords[2], &depthcomm); //k as rank

    double *M, *NT, *P;
    int root_coords[3]={0,0,0}; //The src ==0's node should prepare the data
    MPI_Cart_rank(cartcomm, root_coords, &root); //get the dst
    if(root == me) { //I'm the {0,0,0} node
        initial_matrix(&M, &NT, &P, n, n, n);
    }

    //now each process has
    /*The matrix size is p * p*/
    int i, j, k;
    double *A = (double*)malloc(per_n * per_n * sizeof(double)); //each on has 1
    double *BT = (double*)malloc(per_n * per_n * sizeof(double)); //each only has 1
    double *C = (double*)malloc(per_n * per_n * sizeof(double)); //only per_n * per_n

    //Scatter and Gather used data types
    MPI_Datatype subarrtype;
    int sendcounts[dim_sz*dim_sz];
    int displs[dim_sz*dim_sz]; //value in block (resized) count
    init_subarrtype(root, me, n, dim_sz, per_n, &subarrtype, sendcounts, displs);


    //Send data from root {0, 0, 0} to A{j=0 array}, BT{i=0 array}
    scatter_data(root, me, coords, n, dim_sz, per_n,
            cartcomm, sendcounts, displs, subarrtype,
            A, BT, M, NT);

    mpi_check(MPI_Barrier(cartcomm));
    double t0, t1;
    //Start timing point
    t0 = MPI_Wtime();
    //do broadcast for A/B
    mpi_check(MPI_Bcast(A, per_n*per_n, MPI_DOUBLE, 0, rowcomm));
    mpi_check(MPI_Bcast(BT, per_n*per_n, MPI_DOUBLE, 0, colcomm));


    //do calculation
    for(i = 0; i < per_n; i++) {
        for(j = 0; j < per_n; j++) {
            C[i*per_n+j] = 0;
            for(k = 0; k < per_n; k++) {
                C[i*per_n+j] += A[i*per_n+k] * BT[j*per_n+k];
            }
        }
    }

    // Way 2 - Use MPI_Reduce

    mpi_check(MPI_Reduce(C, C, per_n*per_n, MPI_DOUBLE, MPI_SUM, 0, depthcomm));

    //End timing
    t1 = MPI_Wtime() - t0;
    //node the end time should be in depthcomm with k == 0
    //end timing point
    //use reduction to collect the final time
    MPI_Comm dim_ij_comm;
    MPI_Comm_split(cartcomm, coords[2], coords[0]*dim_sz+coords[1], &dim_ij_comm); //ij as rank


    if(coords[2] == 0) {
        MPI_Reduce(&t1, &t1, 1, MPI_DOUBLE, MPI_SUM, root, dim_ij_comm);
        if(me == root) {
            printf("[mmm3D]P=%d, N=%d, Time=%.9f\n", p, n, t1/(dim_sz*dim_sz));
        }
    }

    gather_result(root, me, coords, n, dim_sz, per_n,
            cartcomm, dim_ij_comm, sendcounts, displs, subarrtype,
            C, M, NT, P);
    MPI_Comm_free(&dim_ij_comm);
    free(A); free(BT); free(C);
    MPI_Type_free(&subarrtype);
    mpi_check(MPI_Finalize());
    return 0;
}
