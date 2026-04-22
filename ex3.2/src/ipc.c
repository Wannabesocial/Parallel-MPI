#include "ipc.h"

#include <mpi.h>
#include <unistd.h>
#include <string.h>

_csr *IPC_csr_create_struct(const int data_size, const int ptr_size){

    _csr *csr;
    int8_t *data;
    int *row_ptr, *indices;

    // Alocate memory
    csr = (_csr *) malloc(sizeof(_csr));
    if(csr == NULL){
        HANDLE_ERROR("malloc 1 -> IPC_csr_create_struct");
    }

    data = (int8_t *) malloc(sizeof(int8_t) * data_size);
    if(data == NULL){
        HANDLE_ERROR("malloc 2 -> IPC_csr_create_struct");
    }

    indices = (int *) malloc(sizeof(int) * data_size);
    if(indices == NULL){
        HANDLE_ERROR("malloc 3 -> IPC_csr_create_struct");
    }

    row_ptr = (int *) malloc(sizeof(int) * ptr_size);
    if(row_ptr == NULL){
        HANDLE_ERROR("malloc 4 -> IPC_csr_create_struct");
    }    
    
    // Save data
    csr->data = data;
    csr->indices = indices;
    csr->row_ptr = row_ptr;
    csr->data_size = data_size;
    csr->ptr_size = ptr_size;

    return csr;
}

_vector *IPC_vec_create_struct(const int size){

    _vector *vec;
    int64_t *array = NULL;

    // Alocate memory 
    vec = (_vector *) malloc(sizeof(_vector));
    if(vec == NULL){
        HANDLE_ERROR("malloc 1 -> IPC_vec_create_struct");
    }

    array = (int64_t *) malloc(sizeof(int64_t) * size);
    if(array == NULL){
        HANDLE_ERROR("malloc 2 -> IPC_vec_create_struct");
    }

    // Save data
    vec->array = array;
    vec->size = size;

    return vec;
}


/* Create partition struct. Alocate memory. Give values */
_partition *IPC_part_create(const int g_rank, const int job_size){

    _partition *part;
    int *send_count, *offsets;
    int offset_sum, work_load, extra_work;

    // Allocate memory
    part = (_partition *) malloc(sizeof(_partition));
    if(part == NULL){
        HANDLE_ERROR("malloc 1 -> IPC_part_create");
    }

    send_count = (int *) malloc(sizeof(int) * g_rank);
    if(send_count == NULL){
        HANDLE_ERROR("malloc 2 -> IPC_part_create");
    }

    offsets = (int *) malloc(sizeof(int) * g_rank);
    if(offsets == NULL){
        HANDLE_ERROR("malloc 3 -> IPC_part_create");
    }

    // Split Work Load
    offset_sum = 0;
    work_load = job_size / g_rank, extra_work = job_size % g_rank;

    for(int i = 0; i < g_rank; i++){
        offsets[i] = offset_sum;
        send_count[i] = (i == 0) ? work_load + extra_work : work_load;
        offset_sum += send_count[i];
    }

    // Save data
    part->send_count = send_count;
    part->offsets = offsets;
    part->size = g_rank;

    return part;
}

/* Destroy partition struct */
void IPC_part_destroy(_partition *part){
    if(part == NULL) return;

    free(part->send_count);
    free(part->offsets);
    free(part);
}


/* ---------------- SEND ---------------------  */

_csr *IPC_send_CSR_Data(const _csr *g_csr, const int g_rank){

    _csr *l_csr;
    int array_size = g_csr->ptr_size-1, work_load, extra_work;
    int start, end, l_data_size, l_ptr_size;
    
    work_load = array_size / g_rank;
    extra_work = array_size % g_rank;

    // Save the MASTER-RANK 0 partial data
    start = 0, end = extra_work + work_load;
    l_csr = IPC_csr_create_struct(g_csr->row_ptr[end], end + 1);

    memcpy(l_csr->data, g_csr->data, g_csr->row_ptr[end] * sizeof(int8_t));
    memcpy(l_csr->indices, g_csr->indices, g_csr->row_ptr[end] * sizeof(int));
    memcpy(l_csr->row_ptr, g_csr->row_ptr, (end + 1) * sizeof(int));

    // Compute sizes that must be send
    for(int i = 1; i < g_rank; i++){

        start = end;
        end += work_load;
        l_data_size = g_csr->row_ptr[end] - g_csr->row_ptr[start];
        l_ptr_size = work_load + 1;

        IPC_send_CSR_Vec_MetaData(l_data_size, l_ptr_size, i);
        MPI_Send(g_csr->data + g_csr->row_ptr[start], l_data_size, MPI_INT8_T, i, 0, MPI_COMM_WORLD);
        MPI_Send(g_csr->indices + g_csr->row_ptr[start], l_data_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(g_csr->row_ptr + start, l_ptr_size, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    return l_csr;
}

void IPC_send_Vec_Data(const _vector *g_vec){
    MPI_Bcast(g_vec->array, g_vec->size, MPI_INT64_T, 0, MPI_COMM_WORLD);
}

void IPC_send_CSR_Vec_MetaData(const int l_csr_data_size, const int l_csr_ptr_size, const int dest){

    int pucket[4 * 2];

    memcpy(pucket    , &l_csr_data_size , 4);
    memcpy(pucket + 4, &l_csr_ptr_size  , 4);
    MPI_Send(pucket, sizeof(pucket), MPI_CHAR, dest, 0, MPI_COMM_WORLD);
}

void IPC_send_MetaData(const int g_vec_size, const int g_mult){

    int pucket[4 * 2];

    memcpy(pucket    , &g_vec_size      , 4);
    memcpy(pucket + 4, &g_mult          , 4);
    MPI_Bcast(pucket, sizeof(pucket), MPI_CHAR, 0, MPI_COMM_WORLD);
}

_IPC_spm *IPC_send_Dense_Data(const _sparse_matrix *spm, const int g_rank){

    _IPC_spm *ipc_spm = NULL;
    _partition *part = NULL;
    int8_t *array, *temp;
    int size = spm->size * spm->size;
    int work_load, extra_work;

    // Save  in 1 Dimension
    array = (int8_t *) malloc(size * sizeof(int8_t));
    if(array == NULL){
        HANDLE_ERROR("malloc -> IPC_send_Dense_Data");
    }
    
    int k = 0;
    for(int i = 0; i < spm->size; i++){
        for(int j = 0; j < spm->size; j++){
            array[k++] = spm->array[i][j];
        }
    }

    work_load = spm->size / g_rank;
    extra_work = spm->size % g_rank;

    // Save MASTER-DATA
    ipc_spm = IPC_spm_create(extra_work + work_load, spm->size);

    part = IPC_part_create(g_rank, spm->size);
    for(int i = 0; i < part->size; i++){
        part->send_count[i] *= spm->size;
        part->offsets[i] *= spm->size;
    }

    temp = (int8_t *) malloc(part->send_count[0] * sizeof(int8_t));
    if(temp == NULL){
        HANDLE_ERROR("malloc -> IPC_send_Dense_Data");
    }    

    // Send to Athers
    MPI_Scatterv(array, part->send_count, part->offsets, MPI_INT8_T,
        temp, part->send_count[0], MPI_INT8_T, 0, MPI_COMM_WORLD);


    // Copy data to Struct
    for(int i = 0; i < ipc_spm->row; i++){
        memcpy(ipc_spm->array[i], temp + (spm->size * i), spm->size);
    }    

    // IPC_spm_display(ipc_spm);

    // DEBUG
    // for(int i = 0; i < size; i++){
    //     if(i % spm->size == 0){
    //         printf("\n");
    //     }

    //     printf("%d\t", array[i]);
    // }
    // printf("\n");


    IPC_part_destroy(part);
    free(array); free(temp);
    return ipc_spm;
}


/* ---------------- RECEAVE --------------------- */

void IPC_recv_CSR_Vec_MetaData(int *l_csr_data_size, int *l_csr_ptr_size){

    int pucket[4 * 2];

    MPI_Recv(pucket, sizeof(pucket), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    memcpy(l_csr_data_size  , pucket    , 4);
    memcpy(l_csr_ptr_size   , pucket + 4, 4);
}

_csr *IPC_recv_CSR_Data(){
    
    _csr *l_csr;
    int l_data_size, l_ptr_size, base;

    // Receave MetaData for struct creation
    IPC_recv_CSR_Vec_MetaData(&l_data_size, &l_ptr_size);
    l_csr = IPC_csr_create_struct(l_data_size, l_ptr_size);

    // Save data
    MPI_Recv(l_csr->data, l_data_size, MPI_INT8_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(l_csr->indices, l_data_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(l_csr->row_ptr, l_ptr_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Fix Offset for Ptr_row
    base = l_csr->row_ptr[0];
    for(int i = 0; i < l_csr->ptr_size; i++){
        l_csr->row_ptr[i] -= base;
    }

    return l_csr;
}

_vector *IPC_recv_Vec_Data(const int l_vec_size){

    _vector *l_vec = IPC_vec_create_struct(l_vec_size);

    MPI_Bcast(l_vec->array, l_vec->size, MPI_INT64_T, 0, MPI_COMM_WORLD);

    return l_vec;
}

void IPC_recv_MetaData(int *g_vec_size, int *g_mult){
    
    int pucket[4 * 2];

    MPI_Bcast(pucket, sizeof(pucket), MPI_CHAR, 0, MPI_COMM_WORLD);
    memcpy(g_vec_size  , pucket     , 4);
    memcpy(g_mult      , pucket + 4 , 4);
}

_IPC_spm *IPC_recv_Dense_Data(const int g_rank, const int l_size){

    _IPC_spm *ipc_spm = NULL;
    int work_load = l_size / g_rank;
    int8_t *array = NULL;

    ipc_spm = IPC_spm_create(work_load, l_size);

    array = (int8_t *) malloc((ipc_spm->col * ipc_spm->row) * sizeof(int8_t));
    if(array == NULL){
        HANDLE_ERROR("malloc -> IPC_send_Dense_Data");
    }

    MPI_Scatterv(NULL, NULL, NULL, MPI_INT8_T,
        array, ipc_spm->col * ipc_spm->row, MPI_INT8_T, 0, MPI_COMM_WORLD);

    // Copy data to Struct
    for(int i = 0; i < ipc_spm->row; i++){
        memcpy(ipc_spm->array[i], array + (l_size * i), l_size);
    }        

    free(array);
    return ipc_spm;
}


/* ----------------COMPUTATIONS ------------------ */

void IPC_matrix_vector_csr(const _csr *l_csr, _vector *l_vec,
    const int l_mult, const int g_rank, int rank)
{
    int size = l_csr->ptr_size - 1;
    int start, end;
    int64_t c, *temp = NULL;
    _partition *part;
    
    part = IPC_part_create(g_rank, l_vec->size);

    temp = (int64_t *) malloc(sizeof(int64_t) * size);
    if(temp == NULL){
        HANDLE_ERROR("malloc -> IPC_matrix_vector_csr");
    }

    for(int t = 0; t < l_mult; t++){

        //for(int k = 0; k < size; k++) temp[k] = 0;
        memset(temp, 0, sizeof(int64_t) * size);

        // Product Calculation
        for(int i = 0; i < size; i++){

            start = l_csr->row_ptr[i];
            end = l_csr->row_ptr[i + 1];
            c = 0;

            for(int j = start; j < end; j++){

                c += l_csr->data[j] * l_vec->array[l_csr->indices[j]];
            }

            temp[i] = c;
        }

        if(t+1 == l_mult){
            MPI_Gatherv(temp, size, MPI_INT64_T,
                l_vec->array, part->send_count, part->offsets, MPI_INT64_T, 0, MPI_COMM_WORLD);
            break;
        }


        // Gather And Send All (the main vector)
        MPI_Allgatherv(temp, size, MPI_INT64_T,
            l_vec->array, part->send_count, part->offsets, MPI_INT64_T, MPI_COMM_WORLD);
    }
    
    IPC_part_destroy(part);
    free(temp);
}

void IPC_matrix_vector_dense(const _IPC_spm *l_spm_dense, _vector *l_vec,
    const int l_mult, const int g_rank)
{
    int size = l_spm_dense->row;
    _partition *part = NULL;
    int64_t c, *temp = NULL;

    part = IPC_part_create(g_rank, l_spm_dense->col);

    temp = (int64_t *) malloc(sizeof(int64_t) * size);
    if(temp == NULL){
        HANDLE_ERROR("malloc -> IPC_matrix_vector_csr");
    }

    //printf("Vec:%d, Col:%d\n", l_vec->size, l_spm_dense->row);

    for(int t = 0; t < l_mult; t++){

        memset(temp, 0, sizeof(int64_t) * size);

        // Compute product
        for(int i = 0; i < l_spm_dense->row; i++){
            c = 0;

            for(int j = 0; j < l_spm_dense->col; j++){
                c += l_vec->array[j] * l_spm_dense->array[i][j];
            }

            temp[i] = c;
        }
        
        if(t+1 == l_mult){
            MPI_Gatherv(temp, size, MPI_INT64_T,
                l_vec->array, part->send_count, part->offsets, MPI_INT64_T, 0, MPI_COMM_WORLD);
            break;
        }

        // Gather All results
        MPI_Allgatherv(temp, size, MPI_INT64_T,
            l_vec->array, part->send_count, part->offsets, MPI_INT64_T, MPI_COMM_WORLD);
    }

    IPC_part_destroy(part);
    free(temp);
}



_IPC_spm *IPC_spm_create(int row, int col){

    _IPC_spm *spm;
    int8_t **array;

    spm = (_IPC_spm *) malloc(sizeof(_IPC_spm));
    if(spm == NULL){
        HANDLE_ERROR("malloc 1 -> IPC_spm_create");
    }

    array = (int8_t **) malloc(sizeof(int8_t *) * row);
    if(array == NULL){
        HANDLE_ERROR("malloc 2 -> IPC_spm_create");
    }
    
    for(int i = 0; i < row; i++){
        array[i] = (int8_t *) malloc(sizeof(int8_t) * col);
        if(array[i] == NULL){
            HANDLE_ERROR("malloc 3 -> IPC_spm_create");
        }
    }

    spm->array = array;
    spm->row = row;
    spm->col = col;

    return spm;
}

void IPC_spm_destroy(_IPC_spm *spm){
    if(spm == NULL) return;

    for(int i = 0; i < spm->row; i++){
        free(spm->array[i]);
    }
    free(spm->array);
    free(spm);
}

void IPC_spm_display(const _IPC_spm *spm){

    printf("Row:%d Col:%d\n", spm->row, spm->col);
    for(int i = 0; i < spm->row; i++){
        for(int j = 0; j < spm->col; j++){
            printf("%d\t", spm->array[i][j]);
        }
        printf("\n");
    }
}



