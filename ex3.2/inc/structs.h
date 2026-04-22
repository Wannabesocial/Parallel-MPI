/**
 * Basic Structs and Funtions for CREATE-DESTROY-DISPLAY
*/

#ifndef STRUCTS_H
#define STRUCTS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SEED 42

/* ------------- ERROR MACROS ---------------- */

#define HANDLE_ERROR(msg) { perror(msg); exit(EXIT_FAILURE); }



/* -------------------- CSR ------------------ */

// Compressed Sparse Row
typedef struct _csr{
    int8_t *data;   // Data storage left->right, top->down
    int *indices;   // Rows that has data
    int *row_ptr;   // What row the 1st none-zero stars, size = size+1
    int data_size;
    int ptr_size;
}_csr;

/* Alocate memory */
_csr *csr_create_struct(const int spm_size, const float zeros_perce);

/* Free allocated memory */
void csr_destroy_struct(_csr *csr);

/* Debug perpuses */
int crs_display(const _csr *csr);

/* -------------- SPARE MATRIX --------------- */

// Sparse Matrix 2D Size x Size
typedef struct _sparse_matrix{
    int8_t **array;
    int size;
}_sparse_matrix;



/* 
Alocate memory.
Initialize with "random" values with respect in zeros%.
Cap in range of CHAR (-128,127) 
*/
_sparse_matrix *spm_create(int size, float zeros_perce);

/* Free allocated memory */
void spm_destroy(_sparse_matrix *spm);

/* Debug perpuses */
int spm_display(const _sparse_matrix *spm);


/* ----------------- Vector ------------------- */

// Vector
typedef struct _vector{
    int64_t *array; // Big loops -> Big values
    int size;
}_vector;

/*
Alocate memory.
Initialize with "random" values.
Cap in range of CHAR (-128,127)
*/
_vector *vec_create(int size);

/* Free allocated memory */
void vec_destroy(_vector *vec);

/* Debug perpuses */
int vec_display(const _vector *vec);

/* Hard copy a vector */
_vector *vec_copy(const _vector *old_vec);

#endif