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


void prepare_data(int dim_sz, int coords[3], int src,
        MPI_Comm cartcomm, MPI_Comm rowcomm, MPI_Comm colcomm, MPI_Comm depthcomm,
        double A[1], double BT[1],
        double* M, double* NT) {
    //now split the data from {0,0,0} -> all
    //use two phase scatter, later may change tot scatter M
    double Mrow[dim_sz];
    double NTrow[dim_sz];
    if (coords[1] == 0 && coords[2] == 0) {
        mpi_check(
                MPI_Scatter(M, dim_sz, MPI_DOUBLE, Mrow, dim_sz, MPI_DOUBLE, src, colcomm));
    }
    if (coords[0] == 0 && coords[2] == 0) {
        mpi_check(
                MPI_Scatter(NT, dim_sz, MPI_DOUBLE, NTrow, dim_sz, MPI_DOUBLE, src, rowcomm));
    }
    mpi_check(MPI_Barrier(cartcomm));
    //now scatter again
    if (coords[1] == 0) {
        mpi_check(
                MPI_Scatter(Mrow, 1, MPI_DOUBLE, A, 1, MPI_DOUBLE, 0, depthcomm));
    }
    if (coords[0] == 0) {
        mpi_check(
                MPI_Scatter(NTrow, 1, MPI_DOUBLE, BT, 1, MPI_DOUBLE, 0, depthcomm));
    }
    mpi_check(MPI_Barrier(cartcomm));
    //finally do bcast
    //now do A's broadcast and B's broadcat
    mpi_check(MPI_Bcast(A, 1, MPI_DOUBLE, 0, rowcomm));
    mpi_check(MPI_Bcast(BT, 1, MPI_DOUBLE, 0, colcomm));

    //    //A
    //    if(coords[1] == 0) { //j == 0
    //        A[0] = coords[2];
    //        //then out
    //        MPI_Send(A, 1, MPI_DOUBLE, nbrs[RIGHT], tag, cartcomm);
    //    } else if( coords[1] < dim_sz - 1) {
    //        MPI_Recv(A, 1, MPI_DOUBLE, nbrs[LEFT], tag, cartcomm, &status);
    //        MPI_Send(A, 1, MPI_DOUBLE, nbrs[RIGHT], tag, cartcomm);
    //    } else { //right most one
    //        MPI_Recv(A, 1, MPI_DOUBLE, nbrs[LEFT], tag, cartcomm, &status);
    //    }
    //    //B
    //    if(coords[0] == 0) { //i ==0
    //        BT[0] = coords[2];
    //        //then out
    //        MPI_Send(BT, 1, MPI_DOUBLE, nbrs[DOWN], tag, cartcomm);
    //    } else if( coords[0] < dim_sz - 1) {
    //        MPI_Recv(BT, 1, MPI_DOUBLE, nbrs[UP], tag, cartcomm, &status);
    //        MPI_Send(BT, 1, MPI_DOUBLE, nbrs[DOWN], tag, cartcomm);
    //    } else { //right most one
    //        MPI_Recv(BT, 1, MPI_DOUBLE, nbrs[UP], tag, cartcomm, &status);
    //    }

}

void gather_result(int dim_sz, int coords[3],
        MPI_Comm rowcomm, MPI_Comm colcomm,
        double C[1],
        double* M, double* NT, double* P) {
    if (coords[2] == 0) { //only the k==0 join the merge
        //all will join
        double CRow[dim_sz];
        mpi_check(
                MPI_Gather(C, 1, MPI_DOUBLE, CRow, 1, MPI_DOUBLE, 0, rowcomm));
        if (coords[1] == 0) { //first column
            mpi_check(
                    MPI_Gather(CRow, dim_sz, MPI_DOUBLE, P, dim_sz, MPI_DOUBLE, 0, colcomm));
            if (coords[0] == 0) { //the root node
                int err_c = check_result(M, NT, P, dim_sz, dim_sz, dim_sz, 0); //not transposed
                if (err_c) {
                    fprintf(stderr, "[MMM3D]Check Failure: %d errors!!!\n", err_c);
                } else {
                    printf("[MMM3D]Result is verified!\n");
                    //print_matrix("C", P, dim_sz, dim_sz, 0);
                }
            }
        }
    }
    //    if(coords[2] != 0) {
    //        //no need the final result shown
    //        MPI_Finalize();
    //        return 0;
    //    }
    //
    //    //firstly just do the j direction gather
    //    dst_coords[2] = 0; //always make k == 0;
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
    int dim_sz = (int)pow(p,1.0/3.0);
    if( dim_sz * dim_sz * dim_sz != p) {
      if (me == 0 ){
          fprintf(stderr, "Process Number %d is not a cubic number!!!\n", p);
      }
      MPI_Finalize();
      exit(1);
    }
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
    int src_coords[3]={0,0,0}; //The src ==0's node should prepare the data
    MPI_Cart_rank(cartcomm, src_coords, &src); //get the dst
    if(src == me) { //I'm the {0,0,0} node
        initial_matrix(&M, &NT, &P, dim_sz, dim_sz, dim_sz);
    }

    //now each process has
    /*The matrix size is p * p*/
    int i, j, k;
    double A[1]; //each on has 1
    double BT[1]; //each only has 1
    double C[1]; //only one

    //Send data from {0,0,0} to all others
    prepare_data(dim_sz, coords, src,
            cartcomm, rowcomm, colcomm, depthcomm,
            A, BT, M, NT);


    //do calculation
    C[0] = A[0] * BT[0];

    // Do reduction along the depth coord
//    // Way 1 - Use explicit send/receive
//    double Buf[0];//used for receive data
//    //now along the k direction reduction
//    if(coords[2] == dim_sz - 1) {
//        MPI_Send(C, 1, MPI_DOUBLE, nbrs[FRONT], tag, cartcomm);
//    } else if(coords[2] > 0){
//        MPI_Recv(Buf, 1, MPI_DOUBLE, nbrs[BACK], tag, cartcomm, &status);
//        C[0] += Buf[0];//reduction
//        MPI_Send(C, 1, MPI_DOUBLE, nbrs[FRONT], tag, cartcomm);
//    } else { //the k == 0
//        MPI_Recv(Buf, 1, MPI_DOUBLE, nbrs[BACK], tag, cartcomm, &status);
//        C[0] += Buf[0];//reduction
//    }

    // Way 2 - Use MPI_Reduce
    MPI_Reduce(C, C, 1, MPI_DOUBLE, MPI_SUM, 0, depthcomm);

    gather_result(dim_sz, coords, rowcomm, colcomm, C, M, NT, P);

    MPI_Finalize();
    return 0;
}
