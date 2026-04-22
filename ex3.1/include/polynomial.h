#ifndef MY_FUN_H
#define MY_FUN_H

#define LCBIT_MASK 0x1  // least significant bit

#define _GNU_SOURCE

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

//  Usefull shits for both SERIAL and PARALLEL implementations

/* Macro to compute the size of the polynomials tha needed base on the power */
#define POLYNOMIAL_SIZE(power) ((power) + 1)

/* Macro to compute the product polynamial size */
#define PRODUCT_POLYNOMIAL_SIZE(power) (2 * (power) + 1)

/* Macro to handle errors (print and exit) */
#define HANDLE_ERROR(msg){  \
    perror(msg);            \
    exit(EXIT_FAILURE);     \
}


/**
 * The representation of the polynomial is in ascending order base on x power
 * +-----+-----+-----+-----+
 * |a1x^0|a2x^1|a3x^2|a4x^3| 3 degree polynomial representation
 * +-----+-----+-----+-----+
 *    0     1     2     3    <-- potition in the array
**/
typedef struct _polynomials{
    char *a;    // first polynomial
    char *b;    // secont polynomial
    int power;  // the power of polynomials
}_polynomials;


/* Return a number in range [-9,9] - {0}. Works with rand,rand_r.
If rand_r pass NULL in pointer */
int valid_coefficient(bool is_rand, unsigned int *seedp);

/* Free polynomials. Destroy what initialized */
void destroy_polynomials(_polynomials *pol);

/* Get the wall cpu time */
double get_time(struct timespec *tic, struct timespec *toc);

/* Check if the two implemetation is equal */
bool is_equal(const int *prod_parallel, const int *prod_serial, int power);

/**
 * Security. Make sure the user give input in right order and is valid.
 * Execute with ./<executable> <loops number> <threads number>
**/
/* Check that given right number of arguments */
void security_user_input(int argc, char **argv);

// ---------------- SERIAL ----------------

/* Create-Malloc the polynomials. Do not forget to free */
_polynomials *serial_creat_polynomials(const int power);

/* Procudt of 2 polynomials */
void serial_polynomial_product(const _polynomials *pol, int *prod_array);


// --------------- PARALLEL (MPI) ---------------

/* Timing structure for MPI operations */
typedef struct _mpi_timings {
    double send_time;       // Time to distribute data to processes
    double compute_time;    // Time for parallel computation
    double receive_time;    // Time to collect results
    double total_time;      // Total execution time (excluding creation)
} mpi_timings;

/* Create-Malloc the polynomials on process 0 only. Do not forget to free */
_polynomials *parallel_creat_polynomials(const int power);

/* Parallel polynomial product using MPI with data distribution */
void parallel_polynomial_product(const _polynomials *pol, int *prod_array, mpi_timings *timings);

#endif
