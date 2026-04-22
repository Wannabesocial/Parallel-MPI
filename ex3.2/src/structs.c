#include "structs.h"
#include <time.h>
#include <string.h>

/* -------------------- CSR ------------------ */

_csr *csr_create_struct(const int spm_size, const float zeros_perce){

    _csr *csr;
    int8_t *data;
    int *row_ptr, *indices;
    int data_size = spm_size * spm_size;
    data_size = data_size - (int)(data_size * zeros_perce);

    csr = (_csr *) malloc(sizeof(_csr));
    if(csr == NULL){
        HANDLE_ERROR("malloc 1 -> csr_create");
    }

    data = (int8_t *) malloc(sizeof(int8_t) * data_size);
    if(data == NULL){
        HANDLE_ERROR("malloc 2 -> csr_create");
    }

    indices = (int *) malloc(sizeof(int) * data_size);
    if(indices == NULL){
        HANDLE_ERROR("malloc 3 -> csr_create");
    }

    row_ptr = (int *) malloc(sizeof(int) * (spm_size + 1));
    if(row_ptr == NULL){
        HANDLE_ERROR("malloc 4 -> csr_create");
    }

    csr->data = data;
    csr->indices = indices;
    csr->row_ptr = row_ptr;
    csr->ptr_size = spm_size + 1;
    csr->data_size = data_size;

    return csr;
}

void csr_destroy_struct(_csr *csr){
    if(csr == NULL) return;

    free(csr->data);
    free(csr->indices);
    free(csr->row_ptr);
    free(csr);
}

int crs_display(const _csr *csr){

    printf("Data Size: %d\n", csr->data_size);
    for(int i = 0; i < csr->data_size; i++){
        printf("%d ", csr->data[i]);
    }
    printf("\n");

    printf("Indices Size: %d\n", csr->data_size);
    for(int i = 0; i < csr->data_size; i++){
        printf("%d ", csr->indices[i]);
    }
    printf("\n");

    printf("Row Ptr Size: %d\n", csr->ptr_size);
    for(int i = 0; i < csr->ptr_size; i++){
        printf("%d ", csr->row_ptr[i]);
    }
    printf("\n");

    return 0;
}

/* -------------- SPARE MATRIX --------------- */

/*
Allocate emmory for 1D array size = size^2
Initialize with 0. And first size*zeros_perce elements with "random".
Cap in range of CHAR (-128,127)
Suffle the result with swaps.
*/
int8_t *spm_create_1D_suffle(int size, float zeros_perce){

    int8_t *array, c;
    int total_size , non_zeros, a, b, temp;

    total_size = size * size;
    non_zeros = total_size - (int)(total_size * zeros_perce);

    // Initialize with 0
    array = (int8_t *) calloc((size * size), sizeof(int8_t));
    if(array == NULL){
        HANDLE_ERROR("calloc -> spm_create")
    }

    // Radnom first "non_zeros" elements
    for(int64_t i = 0; i < non_zeros; i ++){
        
        // Make sure not 0 given
        while((c = rand()) == 0);
        array[i] = c;
    }

    // Traverse the array and constantly swap
    for(int64_t i = 0; i < total_size; i++){

        // Swap between 2 numbers
        a = rand() % total_size;
        b = rand() % total_size;

        temp = array[a];
        array[a] = array[b];
        array[b] = temp;
    }

    return array;
}

_sparse_matrix *spm_create(int size, float zeros_perce){

    _sparse_matrix *spm;
    int8_t **array, *tmp;

    spm = (_sparse_matrix *) malloc(sizeof(_sparse_matrix));
    if(spm == NULL){
        HANDLE_ERROR("malloc 1 -> spm_create");
    }

    array = (int8_t **) malloc(sizeof(int8_t *) * size);
    if(array == NULL){
        HANDLE_ERROR("malloc 2 -> spm_create");
    }

    for(int i = 0; i < size; i++){

        array[i] = (int8_t *) malloc(sizeof(int8_t) * size);
        if(array[i] == NULL){
            HANDLE_ERROR("malloc 3 -> spm_create");
        }
    }

    // Initialize the Spare Matrix
    tmp = spm_create_1D_suffle(size, zeros_perce);
    for(int i = 0; i < size; i++){
        memcpy(array[i], &tmp[i * size], size * sizeof(int8_t));
    }

    free(tmp);

    spm->array = array;
    spm->size = size;

    return spm;
}

void spm_destroy(_sparse_matrix *spm){
    if(spm == NULL) return;

    for(int i = 0; i < spm->size; i++)
        free(spm->array[i]);

    free(spm->array);
    free(spm);
}

int spm_display(const _sparse_matrix *spm){

    printf("Dense Spare Matrix Size: %dx%d\n", spm->size, spm->size);
    for(int i = 0; i < spm->size; i++){
        
        for(int j = 0; j < spm->size; j++){
            printf("%d\t", spm->array[i][j]);
        }

        printf("\n");
    }

    return 0;
}



/* ----------------- Vector ------------------- */

_vector *vec_create(int size){

    _vector *vec;
    int64_t *array;
    int8_t c;

    vec = (_vector *) malloc(sizeof(_vector));
    if(vec == NULL){
        HANDLE_ERROR("malloc 1 -> vec_create");
    }

    array = (int64_t *) malloc(sizeof(int64_t) * size);
    if(array == NULL){
        HANDLE_ERROR("malloc 2 -> vec_create");
    }

    // Initialize vector
    for(int i = 0; i < size; i++){
        
        c = rand(); // [-127, 128]
        array[i] = c;
    }

    vec->array = array;
    vec->size = size;

    return vec;
}

void vec_destroy(_vector *vec){
    if(vec == NULL) return;

    free(vec->array);
    free(vec);
}

int vec_display(const _vector *vec){

    printf("Vector Size: %d\n", vec->size);
    for(int i = 0; i < vec->size; i++){
        printf("%ld ", vec->array[i]);
    }

    printf("\n");
    return 0;
}

_vector *vec_copy(const _vector *old_vec){

    _vector *vec;
    int64_t *array;
    int size = old_vec->size;

    vec = (_vector *) malloc(sizeof(_vector));
    if(vec == NULL){
        HANDLE_ERROR("malloc 1 -> vec_copy");
    }

    array = (int64_t *) malloc(sizeof(int64_t) * size);
    if(array == NULL){
        HANDLE_ERROR("malloc 1 -> vec_copy");
    }

    // Copy data
    memcpy(array, old_vec->array, sizeof(int64_t) * size);
    vec->array = array;
    vec->size = size;

    return vec;
}
