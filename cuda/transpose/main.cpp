#include <cassert>
#include <cstring>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "utils.h"

DECL_TRANSPOSE_FUNC(parallel_transpose_v1);
DECL_TRANSPOSE_FUNC(parallel_transpose_v2);
DECL_TRANSPOSE_FUNC(parallel_transpose_v3);
DECL_TRANSPOSE_FUNC(parallel_transpose_v4);
DECL_TRANSPOSE_FUNC(parallel_transpose_v5);
DECL_TRANSPOSE_FUNC(sequential_transpose_v1);

typedef ull test_type_t;

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Error: need to supply at least one matrix size to benchmark\nUSAGE: %s [N]+\n", argv[0]);
		exit(1);
	}

#ifndef NDEBUG
  int err;
  char host[1024];
  if ((err = gethostname(host, 1024)) != 0) {
    printf("Error getting hostname\n");
  } else {
    printf("Running on host: %s\n", host);
  }
  displayDeviceInfo();
#endif

  const size_t numBenchmarks = argc - 1; 
  size_t Ns[numBenchmarks];
  for (size_t i = 0; i < numBenchmarks; ++i)
    Ns[i] = atoi(argv[i+1]);
  
  Tranposer<test_type_t> tranposing_algos[] = {
    INIT_TRANSPOSER(sequential_transpose_v1, "naive sequential (CPU)"),
    INIT_TRANSPOSER(parallel_transpose_v1, "naive parallel"),
    INIT_TRANSPOSER(parallel_transpose_v2, "32x32 shared mem"),
    INIT_TRANSPOSER(parallel_transpose_v3, "16x16 shared mem"),
    INIT_TRANSPOSER(parallel_transpose_v4, "16x16 tiled"),
    INIT_TRANSPOSER(parallel_transpose_v5, "16x16 tiled, w/o b.c."),
  };

  printf("Benchmarks run on data type of size %d bytes\n", sizeof(test_type_t));
  printf("N\\algo ");
  for (Tranposer<test_type_t> tranposer : tranposing_algos)
    printf("%23s ", tranposer.name.c_str());
  printf("\n");

  for (size_t N : Ns) {
    size_t matrixSize = N * N * sizeof(test_type_t);
    test_type_t *h_mat = (test_type_t *) malloc(matrixSize);
    test_type_t *h_out = (test_type_t *) malloc(matrixSize);
    float transposing_time;

    // make initial array
    for (size_t i = 0; i < N * N; ++i)
      h_mat[i] = rand() % N; 

    printf("%6lu ", N);

    // run each tranpose algo
    for (Tranposer<test_type_t> tranposer : tranposing_algos) {
      std::memset(h_out, 0, matrixSize);
      transposing_time = tranposer.transpose(h_mat, h_out, N);
      cudaDeviceSynchronize(); 

      bool success = check_matrices(h_mat, h_out, N);
      print_test_case_result(success, "%6.2f ms ", transposing_time);
      print_test_case_result(success, "(%6.2f GB/s) ", 2 * matrixSize / transposing_time / 1000000.f );
    }

    printf("\n");

    free(h_out);
    free(h_mat);
  }

  return 0;
}
