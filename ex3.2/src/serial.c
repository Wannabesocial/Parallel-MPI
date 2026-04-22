#include "serial.h"
#include <string.h>

void s_csr_create(const _sparse_matrix *spm, const _csr *csr){

    const int8_t **array = (const int8_t **)spm->array;
    int size = spm->size;
    int idx_d = 0;      // index data-indices
    int non_zeros = 0;  // curent none zero values              

    csr->row_ptr[0] = 0;

    for(int i = 0; i < size; i++){

        for(int j = 0; j < size; j++){
            
            // Skip the zeros elements
            if(array[i][j] == 0)
                continue;
            
            // We have non-zero. Populate struct
            csr->data[idx_d] = array[i][j];
            csr->indices[idx_d] = j;
            idx_d++;
            non_zeros++;
        }

        csr->row_ptr[i+1] = non_zeros;
    }
}

_vector *s_times_matrix_vec_prod_csr(const _csr *csr, const _vector *vec, const char times){

    int64_t *temp, c;
    int size = vec->size;
    int start, end;
    _vector *result;

    result = vec_copy(vec);

    temp = (int64_t *) malloc(sizeof(int64_t) * size);
    if(temp == NULL){
        HANDLE_ERROR("malloc -> s_matrix_vec_prod_spm");
    }

    for(int t = 0; t < times; t++){

        memcpy(temp, result->array, sizeof(int64_t) * size);

        // Product calculation
        for(int i = 0; i < size; i++){
        
            start = csr->row_ptr[i];
            end = csr->row_ptr[i+1];
            c = 0;

            for(int j = start; j < end; j++){

                c += csr->data[j] * result->array[csr->indices[j]];
            }

            temp[i] = c;
        }

        memcpy(result->array, temp, sizeof(int64_t) * size);
    }

    free(temp);
    return result;
}

_vector *s_times_matrix_vec_prod_dense(const _csr *csr, const _vector *vec, const char times){

    int64_t *temp, c;
    int size = vec->size;
    int start, end;
    _vector *result;

    result = vec_copy(vec);

    temp = (int64_t *) malloc(sizeof(int64_t) * size);
    if(temp == NULL){
        HANDLE_ERROR("malloc -> s_matrix_vec_prod_spm");
    }

    for(int t = 0; t < times; t++){

        memcpy(temp, result->array, sizeof(int64_t) * size);

        // Product calculation
        for(int i = 0; i < size; i++){
        
            start = csr->row_ptr[i];
            end = csr->row_ptr[i+1];
            c = 0;

            for(int j = start; j < end; j++){

                c += csr->data[j] * result->array[csr->indices[j]];
            }

            temp[i] = c;
        }

        memcpy(result->array, temp, sizeof(int64_t) * size);
    }

    free(temp);
    return result;
}

