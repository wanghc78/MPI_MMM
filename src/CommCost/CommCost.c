/*
 ============================================================================
 Name        : HelloMPI.c
 Author      : Haichuan Wang
 Version     :
 Copyright   : Haichuan Wang
 Description : Hello MPI World in C 
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include "mpi.h"

/*
 * Try measure the time. Always send first to last and last to first
 * Test different message size. Here we use DOUBLE
 */
int main(int argc, char* argv[]){
	int  me; /* rank of process */
	int  p;       /* number of processes */
	int dst;     /* rank of receiver */
	int tag=0;    /* tag for messages */
    int msg_sz;
    int max_msg_sz = 1024;
    int i; int rep = 1000;
	double data[max_msg_sz];        /* storage for test */
	double time[max_msg_sz];
	MPI_Status status ;   /* return status for receive */

	double t0, t1; //Used for measure the time
	/* start up MPI */
	
	MPI_Init(&argc, &argv);
	
	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	
	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p); 
	
	if( p != 2) {
	    if (me ==0 ){
	      fprintf(stderr, "Process number must be 2 to measure the comm cost!!!\n");
	    }
	    MPI_Finalize();
	    return 1;
	}

	if (me == 0) {
	  dst = p - 1;
	  //at the beginning send some small data to warm up
      for(i = 0; i < rep; i++) MPI_Send(data, 1, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
      for(i = 0; i < rep; i++) MPI_Recv(data, 1, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD, &status);

	  //starting send data
	  for(msg_sz = 1; msg_sz <= max_msg_sz; msg_sz++) {
	    t0 = MPI_Wtime();
	    for(i = 0; i < rep; i++) MPI_Send(data, msg_sz, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
	    for(i = 0; i < rep; i++) MPI_Recv(data, msg_sz, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD, &status);
        t1 = MPI_Wtime();
        time[msg_sz-1] = (t1-t0)/(2.0);
	  }

	  //end print-out result
	  for(msg_sz = 1; msg_sz <= max_msg_sz; msg_sz++) {
	    printf("%d, %.9f\n", msg_sz, time[msg_sz-1]);
	  }

	} else if(me == p - 1) {
	  dst = 0;
	  for(i = 0; i < rep; i++) MPI_Recv(data, 1, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD, &status);
	  for(i = 0; i < rep; i++) MPI_Send(data, 1, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);

	  for(msg_sz = 1; msg_sz <= max_msg_sz; msg_sz++) {
	    for(i = 0; i < rep; i++) MPI_Recv(data, msg_sz, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD, &status);
	    for(i = 0; i < rep; i++) MPI_Send(data, msg_sz, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
      }
	}
	

    /* shut down MPI */
    MPI_Finalize();


    return 0;
}
