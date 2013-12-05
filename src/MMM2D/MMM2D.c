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

void prepare_data(int dim_sz, int coords[2], int src,
        MPI_Comm cartcomm, MPI_Comm rowcomm, MPI_Comm colcomm,
        double A[dim_sz], double BT[dim_sz],
        double* M, double* NT) {
    //just two steps, first scatter, then
    if (coords[1] == 0) { //join then scatter from the column side, M->A
        mpi_check(
                MPI_Scatter(M, dim_sz, MPI_DOUBLE, A, dim_sz, MPI_DOUBLE, src, colcomm));
    }
    if (coords[0] == 0) { //join the scatter from the row side, BT
        mpi_check(
                MPI_Scatter(NT, dim_sz, MPI_DOUBLE, BT, dim_sz, MPI_DOUBLE, src, rowcomm));
    }
    mpi_check(MPI_Barrier(cartcomm));
    //now do A's broadcast and B's broadcat
    mpi_check(MPI_Bcast(A, dim_sz, MPI_DOUBLE, 0, rowcomm));
    mpi_check(MPI_Bcast(BT, dim_sz, MPI_DOUBLE, 0, colcomm));

    //    //A
    //    if(coords[1] == 0) { //j == 0
    //        MPI_Send(A, dim_sz, MPI_DOUBLE, nbrs[RIGHT], tag, cartcomm);
    //    } else if( coords[1] < dim_sz - 1) {
    //        MPI_Recv(A, dim_sz, MPI_DOUBLE, nbrs[LEFT], tag, cartcomm, &status);
    //        MPI_Send(A, dim_sz, MPI_DOUBLE, nbrs[RIGHT], tag, cartcomm);
    //    } else { //right most one
    //        MPI_Recv(A, dim_sz, MPI_DOUBLE, nbrs[LEFT], tag, cartcomm, &status);
    //    }
    //    //B
    //    if(coords[0] == 0) { //i ==0
    //        MPI_Send(BT, dim_sz, MPI_DOUBLE, nbrs[DOWN], tag, cartcomm);
    //    } else if( coords[0] < dim_sz - 1) {
    //        MPI_Recv(BT, dim_sz, MPI_DOUBLE, nbrs[UP], tag, cartcomm, &status);
    //        MPI_Send(BT, dim_sz, MPI_DOUBLE, nbrs[DOWN], tag, cartcomm);
    //    } else { //right most one
    //        MPI_Recv(BT, dim_sz, MPI_DOUBLE, nbrs[UP], tag, cartcomm, &status);
    //    }
}

void gather_result(int dim_sz, int coords[2],
        MPI_Comm rowcomm, MPI_Comm colcomm,
        double C[1],
        double* M, double* NT, double* P) {
    //firstly just do the j direction gather
    //all will join
    double CRow[dim_sz];
    mpi_check(MPI_Gather(C, 1, MPI_DOUBLE, CRow, 1, MPI_DOUBLE, 0, rowcomm));
    if (coords[1] == 0) { //first column
        mpi_check(
                MPI_Gather(CRow, dim_sz, MPI_DOUBLE, P, dim_sz, MPI_DOUBLE, 0, colcomm));
        if (coords[0] == 0) { //the root node
            int err_c = check_result(M, NT, P, dim_sz, dim_sz, dim_sz, 0); //not transposed
            if (err_c) {
                fprintf(stderr, "[MMM2D]Check Failure: %d errors!!!\n", err_c);
            } else {
                printf("[MMM2D]Result is verified!\n");
                //print_matrix("C", P, dim_sz, dim_sz, 0);
            }
        }
    }
    //
    //    if(coords[1] == 0) { //j == 0
    //        double R1[dim_sz];
    //        R1[0] = C[0];
    //        for(j = 1; j < dim_sz; j++) {
    //            dst_coords[0] = coords[0];
    //            dst_coords[1] = j;
    //            MPI_Cart_rank(cartcomm, dst_coords, &dst); //get the dst
    //            MPI_Recv(&R1[j], 1, MPI_DOUBLE, dst, tag, cartcomm, &status); //receive
    //        }
    //        //then i direction gather, only the first column
    //        if(coords[0] == 0) { //i == 0
    //            //copy its own portion
    //            for(j = 0; j < dim_sz; j++) {
    //                P[j] = R1[j];
    //            }
    //            //then receive
    //            for(i = 1; i < dim_sz; i++) {
    //                dst_coords[0] = i;
    //                dst_coords[1] = 0;
    //                MPI_Cart_rank(cartcomm, dst_coords, &dst); //get the dst
    //                MPI_Recv(&P[i*dim_sz], dim_sz, MPI_DOUBLE, dst, tag, cartcomm,  &status); //receive
    //            }
    //
    //            int err_c = check_result(M, NT, P, dim_sz, dim_sz, dim_sz, 0); //not transposed
    //            if(err_c) {
    //                fprintf(stderr, "[Check Failure]%d errors!!!\n", err_c);
    //            } else {
    //                print_matrix("C", P, dim_sz, dim_sz, 0);
    //            }
    //        } else {
    //            dst_coords[0] = 0;
    //            dst_coords[1] = 0;
    //            MPI_Cart_rank(cartcomm, dst_coords, &dst); //get the dst
    //            MPI_Send(R1, dim_sz, MPI_DOUBLE, dst, tag, cartcomm);
    //        }
    //    } else {
    //        dst_coords[0] = coords[0];
    //        dst_coords[1] = 0;
    //        MPI_Cart_rank(cartcomm, dst_coords, &dst); //get the dst
    //        MPI_Send(C, 1, MPI_DOUBLE, dst, tag, cartcomm);
    //    }
}

int main(int argc, char* argv[]) {
    int me; /* rank of process */
    int p; /* number of processes */
    int src; /* rank of sender */
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
      if (me ==0 ){
          fprintf(stderr, "Process Number %d is not a square number!!!\n", p);
      }
      MPI_Finalize();
      exit(1);
    }
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
    int src_coords[2] = {0,0}; //with zero first
    MPI_Cart_rank(cartcomm, src_coords, &src); //get the dst
    if (src == me) {
        initial_matrix(&M, &NT, &P, dim_sz, dim_sz, dim_sz);
    }

    //now each process has
    /*The matrix size is p * p*/
    int i, j, k;
    double A[dim_sz]; //each has one row, i=me, k 0:(p-1)
    double BT[dim_sz]; //j, k
    double C[1]; //only one

    //just two steps, first scatter, then
    prepare_data(dim_sz, coords, src,
            cartcomm,  rowcomm, colcomm,
            A, BT, M, NT);

    //now do the vector vector calculation
    C[0] = 0;
    for(k = 0; k < dim_sz; k++) {
        C[0] += A[k] * BT[k];
    }

    //firstly just do the j direction gather
    //all will join
    gather_result(dim_sz, coords, rowcomm, colcomm, C, M, NT, P);

    MPI_Finalize();
    return 0;
}
