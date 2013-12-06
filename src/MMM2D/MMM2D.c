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


#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

void prepare_data(int n, int per_n, int coords[2], int root,
        MPI_Comm cartcomm, MPI_Comm rowcomm, MPI_Comm colcomm,
        double A[per_n*n], double BT[per_n*n],
        double* M, double* NT) {
    //just two steps, first scatter, then
    if (coords[1] == 0) { //join then scatter from the column side, M->A
        mpi_check(
                MPI_Scatter(M, per_n*n, MPI_DOUBLE, A, per_n*n, MPI_DOUBLE, root, colcomm));
    }
    if (coords[0] == 0) { //join the scatter from the row side, BT
        mpi_check(
                MPI_Scatter(NT, per_n*n, MPI_DOUBLE, BT, per_n*n, MPI_DOUBLE, root, rowcomm));
    }
    mpi_check(MPI_Barrier(cartcomm));

    //The real start point.

    //now do A's broadcast and B's broadcat
    mpi_check(MPI_Bcast(A, per_n*n, MPI_DOUBLE, 0, rowcomm));
    mpi_check(MPI_Bcast(BT, per_n*n, MPI_DOUBLE, 0, colcomm));
}

void gather_result(int n, int per_n, int dim_sz, int coords[2],
        int me, int root,
        MPI_Comm cartcomm, MPI_Comm rowcomm, MPI_Comm colcomm,
        double C[per_n*per_n],
        double* M, double* NT, double* P) {

    //firstly just do the j direction gather
    //In order to get the right gather, I use MPI_Type_vector
    /* create a datatype to describe the subarrays of the global array */

    int sizes[2]    = {n, n};         /* global size */
    int subsizes[2] = {per_n, per_n}; /* local size */
    int starts[2]   = {0,0};          /* where this one starts */
    MPI_Datatype type, subarrtype;
    mpi_check(MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type));
    mpi_check(MPI_Type_create_resized(type, 0, per_n*sizeof(double), &subarrtype));
    mpi_check(MPI_Type_commit(&subarrtype));

    int sendcounts[dim_sz*dim_sz];
    int displs[dim_sz*dim_sz]; //value in block (resized) count
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

    mpi_check(MPI_Gatherv(C, per_n*per_n,  MPI_DOUBLE,
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
    }
    MPI_Type_free(&subarrtype);
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
    int dim_sz = (int)sqrt(p);
    if( dim_sz * dim_sz != p) {
      if (me == root){
          fprintf(stderr, "Process Number %d is not a square number!!!\n", p);
      }
      MPI_Finalize();
      exit(1);
    }

    int n = get_problem_size(argc, argv, dim_sz, me);
    int per_n = n / dim_sz;

    //prepare for the cartesian topology
    MPI_Comm cartcomm;
    int nbrs[4], dims[2] = {dim_sz, dim_sz};
    int periods[2] = {0,0}, reorder=0,coords[2];
    mpi_check(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm));
    MPI_Comm_rank(cartcomm, &me);
    MPI_Cart_coords(cartcomm, me, 2, coords);
    MPI_Cart_shift(cartcomm, 0, 1, &nbrs[UP], &nbrs[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);


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
    double *A = (double*)malloc(per_n * n*sizeof(double)); //each has per_n row, i=me, k 0:(p-1)
    double *BT = (double*)malloc(per_n * n*sizeof(double)); //j, k
    double *C = (double*)malloc(per_n * per_n * sizeof(double));

    //just two steps, first scatter, then
    prepare_data(n, per_n, coords, root,
            cartcomm,  rowcomm, colcomm,
            A, BT, M, NT);

    //now do the vector vector calculation
    for(i = 0; i < per_n; i++) {
        for(j = 0; j < per_n; j++) {
            C[i*per_n+j] = 0;
            for(k = 0; k < n; k++) {
                C[i*per_n+j] += A[i*n+k] * BT[j*n+k];
            }
        }
    }

    //firstly just do the j direction gather
    //all will join
    gather_result(n, per_n, dim_sz, coords, me, root,
            cartcomm, rowcomm, colcomm, C, M, NT, P);

    free(A); free(BT); free(C);
    MPI_Finalize();
    return 0;
}
