/**
 * Serial Algorithms Implementation
 */

#ifndef SERIAL_H
#define SERIAL_H

#include "structs.h"
#include <stdbool.h>

/* Create Compressed Sparse Row base on a Sparse Matrix */
void s_csr_create(const _sparse_matrix *spm, const _csr *csr);

/* Serial CSR-Vector Product */
_vector *s_times_matrix_vec_prod_csr(const _csr *csr, const _vector *vec, const char times);

/* Serial Dense-Vector Product */
_vector *s_times_matrix_vec_prod_dense(const _csr *csr, const _vector *vec, const char times);

#endif