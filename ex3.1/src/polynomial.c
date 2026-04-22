#include "polynomial.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char **argv){

    int rank, size;

    /* Start up MPI */
    MPI_Init(&argc, &argv);

    /* Get the number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Get my rank among all the processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    security_user_input(argc, argv);

    /* User input values (argc, argv) */
    int power = atoi(argv[1]);

    int *prod_parallel;
    struct timespec tic, toc;
    mpi_timings timings = {0};

    /* Synchronize all processes before starting timing */
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        clock_gettime(CLOCK_MONOTONIC, &tic);
    }
    _polynomials *pol = parallel_creat_polynomials(power);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        clock_gettime(CLOCK_MONOTONIC, &toc);
        printf("Create polynomials %.6f sec\n", get_time(&tic, &toc));
    }

    // Create space for parallel implementation (only rank 0 needs final result)
    if(rank == 0){
        prod_parallel = (int *) calloc(PRODUCT_POLYNOMIAL_SIZE(power), sizeof(int));
        if(prod_parallel == NULL){
            HANDLE_ERROR("In function (main), calloc(prod_parallel)");
        }
    } else {
        prod_parallel = (int *) calloc(PRODUCT_POLYNOMIAL_SIZE(power), sizeof(int));
        if(prod_parallel == NULL){
            HANDLE_ERROR("In function (main), calloc(prod_parallel)");
        }
    }

    /* Parallel computation with detailed timing */
    parallel_polynomial_product(pol, prod_parallel, &timings);

    if(rank == 0){
        printf("Send data time %.6f sec\n", timings.send_time);
        printf("Compute time %.6f sec\n", timings.compute_time);
        printf("Receive data time %.6f sec\n", timings.receive_time);
        printf("Total execution time %.6f sec (with %d processes)\n", timings.total_time, size);
    }

    free(prod_parallel);
    destroy_polynomials(pol);

    /* Shut down MPI */
    MPI_Finalize();

    return 0;
}
