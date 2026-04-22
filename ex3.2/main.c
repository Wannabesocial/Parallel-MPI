//#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>     /* For MPI functions, etc */ 

#include "structs.h"
#include "serial.h"
#include "ipc.h"
#include "valid.h"


/**
 * g = global variables
 * l = local variables
 */

int main(int argc, char **argv){

   valid_arguments(argc, (const char **)argv); // Make sure user give right input

   /* MPI */
   int g_rank, l_rank;
   char hostname[20];

   /* Global-Master variables type shit */
   int g_size = atoi(argv[1]), g_mult = atoi(argv[3]);
   float g_sparsity = atoi(argv[2]) / 100.0f;

   _vector *g_vec = NULL,*vec_dense_result = NULL, *vec_csr_result = NULL;
   _vector *vec_s_csr = NULL, *vec_s_dense = NULL; // Serial results
   _csr *g_csr = NULL;
   _sparse_matrix *g_spm = NULL;
   _IPC_spm *l_spm = NULL;

   /* Time structs */
   float total_csr = 0, x, total_dense = 0;
   struct timespec tic, toc;


   /* Local-Per Proccess variables type shit (Master has it 2) */
   int l_size = 0, l_mult = 0;
   _vector *l_vec = NULL, *l_vec_dense = NULL;
   _csr *l_csr = NULL;


   MPI_Init(NULL, NULL);                    /* Start up MPI */
   MPI_Comm_size(MPI_COMM_WORLD, &g_rank);  /* Get the number of processes */ 
   MPI_Comm_rank(MPI_COMM_WORLD, &l_rank);  /* Get my rank among all the processes */
   
   gethostname(hostname, 20);

   // I am the MASTER OF DESASTER
   if(l_rank == 0){

      // Make Random Vector and Matrix with respect in Sparsity
      srand(SEED);
      g_spm = spm_create(g_size, g_sparsity);
      g_vec = vec_create(g_size);

      clock_gettime(CLOCK_MONOTONIC, &tic);
      g_csr = csr_create_struct(g_size, g_sparsity);
      s_csr_create(g_spm, g_csr);
      clock_gettime(CLOCK_MONOTONIC, &toc);
      x = (toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1e9;
      total_csr += x;
      printf("CSR Create time: %.6f\n", x);


      // Serial--Bleed from MPI structurs 
      if(g_rank == 1){

         clock_gettime(CLOCK_MONOTONIC, &tic);
         vec_s_csr = s_times_matrix_vec_prod_csr(g_csr, g_vec, g_mult);
         clock_gettime(CLOCK_MONOTONIC, &toc);
         x = (toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1e9;
         total_csr += x;
         printf("CSR-Vector Serial time: %.6f\n", x);

         clock_gettime(CLOCK_MONOTONIC, &tic);
         vec_s_dense = s_times_matrix_vec_prod_dense(g_csr, g_vec, g_mult);
         clock_gettime(CLOCK_MONOTONIC, &toc);
         printf("Dense-Vector Serial time: %.6f\n", (toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1e9);
         
         printf("Total time CSR: %.6f\n", total_csr);

         printf("=========================\n");
         printf("Serial CSR and Dense: ");
         valid_vectors(vec_s_csr, vec_s_dense) ? printf("SUCCESS\n") : printf("FAIL\n");

         goto clean_up;
      }


      IPC_send_MetaData(g_vec->size, g_mult);

      clock_gettime(CLOCK_MONOTONIC, &tic);
      l_csr = IPC_send_CSR_Data(g_csr, g_rank);
      clock_gettime(CLOCK_MONOTONIC, &toc);
      x = (toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1e9;
      total_csr += x;
      printf("CSR Send time: %.6f\n", x);
      
      IPC_send_Vec_Data(g_vec);

      vec_csr_result = vec_copy(g_vec);

      clock_gettime(CLOCK_MONOTONIC, &tic);
      IPC_matrix_vector_csr(l_csr, vec_csr_result, g_mult, g_rank, l_rank);
      clock_gettime(CLOCK_MONOTONIC, &toc);
      x = (toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1e9;
      total_csr += x;
      printf("CSR-Vector Parallel time: %.6f\n", x);

      clock_gettime(CLOCK_MONOTONIC, &tic);
      l_spm = IPC_send_Dense_Data(g_spm, g_rank);
      clock_gettime(CLOCK_MONOTONIC, &toc);
      x = (toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1e9;
      total_dense += x;
      printf("Dense Send time: %.6f\n", x);

      IPC_send_Vec_Data(g_vec);

      vec_dense_result = vec_copy(g_vec);

      clock_gettime(CLOCK_MONOTONIC, &tic);
      IPC_matrix_vector_dense(l_spm, vec_dense_result, g_mult, g_rank);
      clock_gettime(CLOCK_MONOTONIC, &toc);
      x = (toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) / 1e9;
      total_dense += x;
      printf("Dense-Vector Parallel time: %.6f\n", x);
      
      printf("Total CSR-Vector Time:%.6f\n", total_csr);
      printf("Total Dense-Vector Time:%.6f\n", total_dense);

      printf("==========================\n");
      printf("Parallel CSR and Dense Result:");
      valid_vectors(vec_csr_result, vec_dense_result) ? printf("SUCCESS\n"): printf("FAIL\n");
   }
   else{

      IPC_recv_MetaData(&l_size, &l_mult);
      l_csr = IPC_recv_CSR_Data();
      l_vec = IPC_recv_Vec_Data(l_size);

      IPC_matrix_vector_csr(l_csr, l_vec, l_mult, g_rank, l_rank);

      l_spm = IPC_recv_Dense_Data(g_rank, l_size);
      l_vec_dense = IPC_recv_Vec_Data(l_size);
      IPC_matrix_vector_dense(l_spm, l_vec_dense, l_mult, g_rank);

   }

clean_up:
   // Clean UP
   spm_destroy(g_spm);
   vec_destroy(g_vec);
   csr_destroy_struct(g_csr);

   vec_destroy(vec_s_csr);
   vec_destroy(vec_s_dense);

   vec_destroy(l_vec);
   vec_destroy(l_vec_dense);
   vec_destroy(vec_dense_result);
   vec_destroy(vec_csr_result);

   csr_destroy_struct(l_csr);
   IPC_spm_destroy(l_spm);

   MPI_Finalize();   /* Shut down MPI */
   return 0;
}