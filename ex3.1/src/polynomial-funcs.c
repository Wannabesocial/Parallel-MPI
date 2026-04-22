#include "polynomial.h"

#include <time.h>
#include <mpi.h>
#include <unistd.h>


int valid_coefficient(bool is_rand, unsigned int *seedp){

    int number;

    /* Generate only non-zero positive integers (1-9) */
    number = (is_rand) ? rand() : rand_r(seedp);
    number = number % 9 + 1;

    return number;
}

void destroy_polynomials(_polynomials *pol){
    if(pol != NULL){
        if(pol->a != NULL) free(pol->a);
        if(pol->b != NULL) free(pol->b);
        free(pol);
    }
}

double get_time(struct timespec *tic, struct timespec *toc){
    return (toc->tv_sec - tic->tv_sec) + (toc->tv_nsec - tic->tv_nsec) / 1e9;
}

bool is_equal(const int *prod_parallel, const int *prod_serial, int power){

    for(int i = 0; i < PRODUCT_POLYNOMIAL_SIZE(power); i++){
        if(prod_parallel[i] != prod_serial[i]){
            return false;
        }
    }

    return true;
}

void security_user_input(int argc, char **argv){

    if(argc != 2){
        HANDLE_ERROR("Usage: mpiexec -n <num_processes> ./<executable> <polynomial_power>");
    }

    int power = atoi(argv[1]);

    if(power <= 0){
        HANDLE_ERROR("Polynomial power must be positive");
    }
}


_polynomials *serial_creat_polynomials(const int power){

    srand(time(NULL));

    _polynomials *pol = (_polynomials *) malloc(sizeof(_polynomials));
    if(pol == NULL){
        HANDLE_ERROR("In function (serial_creat_polynomials), malloc _polynomials");
    }

    char *a = (char *) malloc(POLYNOMIAL_SIZE(power) * sizeof(char));
    if(a == NULL){
        HANDLE_ERROR("In function (serial_creat_polynomials), malloc a");
    }

    char *b = (char *) malloc(POLYNOMIAL_SIZE(power) * sizeof(char));
    if(b == NULL){
        HANDLE_ERROR("In function (serial_creat_polynomials), malloc b");
    }

    /* Initialization */
    for(int i = 0; i < POLYNOMIAL_SIZE(power); i++){
        a[i] = valid_coefficient(true, NULL);
        b[i] = valid_coefficient(true, NULL);
    }

    pol->a = a;
    pol->b = b;
    pol->power = power;

    return pol;
}

void serial_polynomial_product(const _polynomials *pol, int *prod_array){


    int magic_size = 1000;
    int komati = POLYNOMIAL_SIZE(pol->power) / magic_size;
    int last;

    for(int i = 0; i < POLYNOMIAL_SIZE(pol->power); i++){

        for(int k = 0; k < magic_size; k++){

            last = (k+1 == magic_size) ? POLYNOMIAL_SIZE(pol->power) : k*komati+komati;

            for(int j = k*komati; j < last; j++){
                prod_array[i + j] += pol->a[i] * pol->b[j];
            }
        }
    }
}


_polynomials *parallel_creat_polynomials(const int power){

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    _polynomials *pol = (_polynomials *) malloc(sizeof(_polynomials));
    if(pol == NULL){
        HANDLE_ERROR("In function (parallel_creat_polynomials), malloc _polynomials");
    }

    pol->power = power;

    /* Only process 0 creates the polynomials */
    if(rank == 0){
        int poly_size = POLYNOMIAL_SIZE(power);
        unsigned int base_seed = time(NULL);
        unsigned int seedp;

        char *a = (char *) malloc(poly_size * sizeof(char));
        if(a == NULL){
            HANDLE_ERROR("In function (parallel_creat_polynomials), malloc a");
        }

        char *b = (char *) malloc(poly_size * sizeof(char));
        if(b == NULL){
            HANDLE_ERROR("In function (parallel_creat_polynomials), malloc b");
        }

        /* Generate polynomial a with non-zero positive coefficients */
        seedp = base_seed;
        for(int i = 0; i < poly_size; i++){
            a[i] = valid_coefficient(false, &seedp);
        }

        /* Generate polynomial b with non-zero positive coefficients */
        seedp = base_seed + 1000;
        for(int i = 0; i < poly_size; i++){
            b[i] = valid_coefficient(false, &seedp);
        }

        pol->a = a;
        pol->b = b;
    } else {
        /* Other processes don't allocate or create polynomials */
        pol->a = NULL;
        pol->b = NULL;
    }

    return pol;
}

void parallel_polynomial_product(const _polynomials *pol, int *prod_array, mpi_timings *timings){

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int poly_size = POLYNOMIAL_SIZE(pol->power);
    int prod_size = PRODUCT_POLYNOMIAL_SIZE(pol->power);

    struct timespec total_tic, total_toc, send_tic, send_toc;
    struct timespec compute_tic, compute_toc, recv_tic, recv_toc;
    double send_time = 0.0, receive_time = 0.0;

    /* Start total timing */
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) clock_gettime(CLOCK_MONOTONIC, &total_tic);

    /* Calculate chunk sizes for each process */
    int chunk_size = poly_size / size;
    int remainder = poly_size % size;

    int *sendcounts = NULL;
    int *displs = NULL;
    char *local_a = NULL;
    char *b_buffer = NULL;

    if(rank == 0){
        sendcounts = (int *) malloc(size * sizeof(int));
        displs = (int *) malloc(size * sizeof(int));

        for(int i = 0; i < size; i++){
            sendcounts[i] = chunk_size + (i < remainder ? 1 : 0);
            displs[i] = i * chunk_size + (i < remainder ? i : remainder);
        }

        clock_gettime(CLOCK_MONOTONIC, &send_tic);
    }

    /* Determine local chunk size for this process */
    int local_chunk_size = chunk_size + (rank < remainder ? 1 : 0);
    local_a = (char *) malloc(local_chunk_size * sizeof(char));
    if(local_a == NULL){
        HANDLE_ERROR("In function (parallel_polynomial_product), malloc local_a");
    }

    /* Allocate space for polynomial b on all processes */
    b_buffer = (char *) malloc(poly_size * sizeof(char));
    if(b_buffer == NULL){
        HANDLE_ERROR("In function (parallel_polynomial_product), malloc b_buffer");
    }

    /* Process 0 distributes data to all processes */
    if(rank == 0){
        for(int i = 0; i < poly_size; i++){
            b_buffer[i] = pol->b[i];
        }
    }
    MPI_Bcast(b_buffer, poly_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    /* Scatter chunks of polynomial a to all processes */
    MPI_Scatterv(rank == 0 ? pol->a : NULL, sendcounts, displs, MPI_CHAR,
                 local_a, local_chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if(rank == 0){
        clock_gettime(CLOCK_MONOTONIC, &send_toc);
        send_time = get_time(&send_tic, &send_toc);
    }

    /* All processes compute their local product */
    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &compute_tic);

    int *local_prod = (int *) calloc(prod_size, sizeof(int));
    if(local_prod == NULL){
        HANDLE_ERROR("In function (parallel_polynomial_product), calloc local_prod");
    }

    /* Calculate starting index for this process's chunk */
    int start_idx = rank * chunk_size + (rank < remainder ? rank : remainder);

    /* Compute local product: for each coefficient in local_a, multiply with all of b */
    int magic_size = 1000;
    int komati = poly_size / magic_size;
    int last;

    for(int i = 0; i < local_chunk_size; i++){
        int global_i = start_idx + i;
        for(int k = 0; k < magic_size; k++){
            last = (k+1 == magic_size) ? poly_size : k*komati+komati;
            for(int j = k*komati; j < last; j++){
                local_prod[global_i + j] += local_a[i] * b_buffer[j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &compute_toc);
    double compute_time = get_time(&compute_tic, &compute_toc);

    /* Process 0 collects results */
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        clock_gettime(CLOCK_MONOTONIC, &recv_tic);
    }

    MPI_Reduce(local_prod, prod_array, prod_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        clock_gettime(CLOCK_MONOTONIC, &recv_toc);
        receive_time = get_time(&recv_tic, &recv_toc);
    }

    /* End total timing */
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        clock_gettime(CLOCK_MONOTONIC, &total_toc);
        timings->send_time = send_time;
        timings->compute_time = compute_time;
        timings->receive_time = receive_time;
        timings->total_time = get_time(&total_tic, &total_toc);
    }

    /* Cleanup */
    free(local_prod);
    free(local_a);
    free(b_buffer);
    if(rank == 0){
        free(sendcounts);
        free(displs);
    }
}
