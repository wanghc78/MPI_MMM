/*
 * utilith.h
 *
 *  Created on: Nov 12, 2013
 *      Author: Haichuan Wang
 */

#ifndef UTILITH_H_
#define UTILITH_H_

#include <mpi.h>

#define mpi_check(stmt) do {                               \
        int err = stmt;                                    \
        if (err != MPI_SUCCESS) {                          \
            fprintf(stderr, "MPI_ERROR %d: from running stmt %s\n", err ,#stmt); \
            MPI_Finalize(); \
            exit(err);  \
        }                                                  \
    } while(0)


/*
 * According to the input to decide the problem size
 */
int get_problem_size(int argc, char* argv[], int p, int me);

void init_subarrtype(int root, int me,
        int n, int dim_sz, int per_n,
        MPI_Datatype* subarrtype_addr, int sendcounts[], int displs[]);

/*
 * Use random data to initialize the
 * A[m*c], B[c*n]/BT[n*c]  C[m*n]
 */
void initial_matrix(double** A_addr, double** BT_addr, double** C_addr, int m, int n, int c);


/*
 * do a local calculation and compare it with C.
 * transposed: whether C is transposed (row order, or column order)
 */
int check_result(double* A, double* BT, double* C, int m, int n, int c, int transposed);

void print_matrix(char* name, double* A, int m, int n, int transposed);

void free_matrix(double* A, double* B, double *C);


#endif /* UTILITH_H_ */
