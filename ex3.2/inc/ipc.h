/**
 * Wrapper funtions for MPI. + Functions for working structs
 */

#ifndef IPC_H
#define IPC_H

#include "structs.h"

/* Spare Matrix Extention for MPI spilt */
typedef struct _IPC_spm{
    int8_t **array;
    int row;
    int col;
}_IPC_spm;

/* Allocate memory */
_IPC_spm *IPC_spm_create(int row, int col);

/* Free memory */
void IPC_spm_destroy(_IPC_spm *spm);

/* For Debuging */
void IPC_spm_display(const _IPC_spm *spm);



/* Helpful to compute send 'send counts' and 'offsets' */
typedef struct _partition{
    int *send_count;
    int *offsets;
    int size;
}_partition;

/* MPI extantion -> 'data_size' and 'ptr_size' is known.
Just alocate memory */
_csr *IPC_csr_create_struct(const int data_size, const int ptr_size);


/* MPI extantion -> 'data_size' is known.
Just alocate memory */
_vector *IPC_vec_create_struct(const int size);


/* ---------------- SEND ---------------------  */

/* Send MetaData(sizes etc.) for CSR-Vector Product */
void IPC_send_CSR_Vec_MetaData(const int l_csr_data_size, const int l_csr_ptr_size, const int dest);

/* Send 'Compressed Sparse Row' data. Master-Caller.
Save Master WorkLoad to 'l_csr' */
_csr *IPC_send_CSR_Data(const _csr *g_csr, const int g_rank);

/* Send 'Vector' Data -- BroadCast */
void IPC_send_Vec_Data(const _vector *g_vec);

/* Send Meta data aka User Input */
void IPC_send_MetaData(const int g_vec_size, const int g_mult);

/* Send Data for Matrix-Vector product Dense */
_IPC_spm *IPC_send_Dense_Data(const _sparse_matrix *spm, const int g_rank);


/* ---------------- RECEAVE --------------------- */

/* Receave MetaData(sizes etc.) */
void IPC_recv_CSR_Vec_MetaData(int *l_csr_data_size, int *l_csr_ptr_size);

/* Receave 'Compressed Sparse Row' data */
_csr *IPC_recv_CSR_Data();

/* Receave 'Vector' data. */
_vector *IPC_recv_Vec_Data(const int l_vec_size);

/* Receave Meta data aka User Input */
void IPC_recv_MetaData(int *g_vec_size, int *g_mult);

/* Receave 'Dense Matrix' data */
_IPC_spm *IPC_recv_Dense_Data(const int g_rank, const int l_size);


/* ----------------COMPUTATIONS ------------------ */

/* Compute CSR-Vector Product */
void IPC_matrix_vector_csr(const _csr *l_csr, _vector *l_vec,
    const int l_mult, const int g_rank, int rank);

/*  */
void IPC_matrix_vector_dense(const _IPC_spm *l_spm_dense, _vector *l_vec,
    const int l_mult, const int g_rank);


#endif