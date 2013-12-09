/*
 ============================================================================
 Name        : MMM25D.c
 Author      : Haichuan Wang
 Version     :
 Copyright   : Your copyright notice
 Description : 2.5D Algorithm - Communication-optimal parallel 2.5D matrix multiplication and
               LU factorization algorithms
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
        MPI_Comm dim_ij_comm, int sendcounts[p_gridsize*p_gridsize],  int displs[p_gridsize*p_gridsize], MPI_Datatype subarrtype,
        double A[n_local*n_local], double BT[n_local*n_local],
        double* M, double* NT) {

    if(coords[2] == 0) {
        mpi_check(MPI_Scatterv(M, sendcounts,  displs, subarrtype,
                A, n_local*n_local, MPI_DOUBLE, root, dim_ij_comm));
    }


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

    if(coords[2] == 0) {
        mpi_check(MPI_Scatterv(NT, sendcounts, displsBT, subarrtype,
                BT, n_local*n_local, MPI_DOUBLE, root, dim_ij_comm));
    }
}

void gather_result(int root, int me, int coords[3],
        int n, int p_gridsize, int n_local,
        MPI_Comm dim_ij_comm, int sendcounts[p_gridsize*p_gridsize],  int displs[p_gridsize*p_gridsize], MPI_Datatype subarrtype,
        double C[n_local*n_local],
        double* M, double* NT, double* P) {

    if(coords[2] == 0) {
        mpi_check(MPI_Gatherv(C, n_local*n_local,  MPI_DOUBLE,
                     P, sendcounts, displs, subarrtype,
                     root, dim_ij_comm));
    }


    //all in root now
    if(me == root) {
        int err_c = check_result(M, NT, P, n, n, n, 0); //not transposed
        if (err_c) {
            fprintf(stderr, "[MMM25D]Check Failure: %d errors\n", err_c);
            //print_matrix("P", P, n, n, 0);
        } else {
            printf("[MMM52D]Result is verified\n");
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

    /*p_gridsize * p_gridsize * c = p */
    /* c is in argv[2]*/
    int c, p_gridsize;
    if(argc > 2) {
        c = atoi(argv[2]);
        p_gridsize = (int)sqrt(p/c);
    } else {
        p_gridsize = (int)sqrt(p/2);
        c = p/(p_gridsize * p_gridsize);
    }
    if( p_gridsize * p_gridsize * c != p) {
      if (me == root){
          fprintf(stderr, "[ERROR]P=%d, the chosen c=%d, cannot decide p_gridsize to meet c*(p_gridsize)^2=p!!\n", p, c);
      }
      MPI_Finalize();
      exit(1);
    }
    if( c > p_gridsize ) {
        if (me == root){
            fprintf(stderr, "[ERROR]c=%d is larger than p_gridsize=%d!\n", c, p_gridsize);
        }
        MPI_Finalize();
        exit(1);
    }

    int n = get_problem_size(argc, argv, p_gridsize, me);
    int n_local = n / p_gridsize;
    if(me == root) {printf("[MMM25D]P=%d, n=%d, p_gridsize=%d, c=%d\n", p, n, p_gridsize, c);}

    //prepare for the cartesian topology
    MPI_Comm cartcomm;
    int dims[3] = {p_gridsize, p_gridsize, c};
    int periods[3] = {1,1,0}, reorder=0,coords[3];
    mpi_check(MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cartcomm));
    MPI_Comm_rank(cartcomm, &me);
    MPI_Cart_coords(cartcomm, me, 3, coords);



    //Split the cartcomm into rowcomm, colcomm
    MPI_Comm rowcomm, colcomm, depthcomm;
    MPI_Comm_split(cartcomm, coords[0]*p_gridsize+coords[2], coords[1], &rowcomm); //j as rank
    MPI_Comm_split(cartcomm, coords[1]*p_gridsize+coords[2], coords[0], &colcomm); //i as rank
    MPI_Comm_split(cartcomm, coords[0]*p_gridsize+coords[1], coords[2], &depthcomm); //k as rank

    MPI_Comm dim_ij_comm; //used for scatter data and collect result
    MPI_Comm_split(cartcomm, coords[2], coords[0]*p_gridsize+coords[1], &dim_ij_comm); //ij as rank



    double *M, *NT, *P;
    int root_coords[3] = {0,0,0}; //with zero first
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

    //Now A/BT local are ready accorss layer 0.
    scatter_data(root, me, coords, n, p_gridsize, n_local,
            dim_ij_comm, sendcounts, displs, subarrtype,
            A, BT, M, NT);

    mpi_check(MPI_Barrier(cartcomm));
    double t0, t1;
    //Start timing point
    t0 = MPI_Wtime();
    //the main MMM25D algorithm part
    //The real start point.
    mpi_check(MPI_Bcast(A, n_local*n_local, MPI_DOUBLE, 0, depthcomm));
    mpi_check(MPI_Bcast(BT,n_local*n_local, MPI_DOUBLE, 0, depthcomm));

    //first need shift both A and B.
    int dst, src; //use for shift
    int datatag = 0;
    //For the i-th row of the subtasks grid the matrix  A blocks are shifted for (i-1) positions to the left,
    int shift = coords[0] + coords[2] * p_gridsize / c;
    if(shift > 0) {
        dst = (coords[1] +  c*p_gridsize - shift) % p_gridsize;
        src = (coords[1] + shift) % p_gridsize;

        MPI_Sendrecv_replace(A, n_local*n_local, MPI_DOUBLE, dst,
        datatag, src, datatag, rowcomm, &status);
    }

    shift = coords[1] + coords[2] * p_gridsize / c;
    if(shift > 0) {
        dst = (coords[0] +  c*p_gridsize - shift) % p_gridsize;
        src = (coords[0] + shift) % p_gridsize;

        MPI_Sendrecv_replace(BT, n_local*n_local, MPI_DOUBLE, dst,
        datatag, src, datatag, colcomm, &status);
    }

    mpi_check(MPI_Barrier(cartcomm));
    //finish initial shift


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

    for(stage++; stage < p_gridsize/c; stage++) {
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

    //now do depth reduction
    mpi_check(MPI_Reduce(C, C, n_local*n_local, MPI_DOUBLE, MPI_SUM, 0, depthcomm));

    //End timing
    t1 = MPI_Wtime() - t0;
    //end timing point
    //use reduction to collect the final time
    if(coords[2] == 0) {
        MPI_Reduce(&t1, &t1, 1, MPI_DOUBLE, MPI_SUM, root, dim_ij_comm);
        if(me == root) {
            printf("[mmm25D]P=%d, N=%d, C=%d, Time=%.9f\n", p, n, c, t1/(p_gridsize*p_gridsize));
        }
    }
#ifdef VERIFY
    gather_result(root, me, coords, n, p_gridsize, n_local,
            dim_ij_comm, sendcounts, displs, subarrtype,
            C, M, NT, P);
#endif
    MPI_Type_free(&subarrtype);
    MPI_Comm_free(&dim_ij_comm);
    mkl_free(A); mkl_free(BT); mkl_free(C);
    MPI_Finalize();
    return 0;
}
