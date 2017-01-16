#include <cassert>
#include <cstring>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

DECL_TRANSPOSE_FUNC(parallel_transpose_v1);
DECL_TRANSPOSE_FUNC(parallel_transpose_v2);
DECL_TRANSPOSE_FUNC(parallel_transpose_v3);
DECL_TRANSPOSE_FUNC(parallel_transpose_v4);
DECL_TRANSPOSE_FUNC(parallel_transpose_v5);
DECL_TRANSPOSE_FUNC(sequential_transpose_v1);

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Error: need to supply at least one matrix size to benchmark\nUSAGE: %s [N]+\n", argv[0]);
		exit(1);
	}

  displayDeviceInfo();

  const size_t numBenchmarks = argc - 1; 
  size_t Ns[numBenchmarks];
  for (size_t i = 0; i < numBenchmarks; ++i)
    Ns[i] = atoi(argv[i+1]);
  
  Tranposer<ull> tranposing_algos[] = {
    INIT_TRANSPOSER(sequential_transpose_v1, "naive sequential"),
    INIT_TRANSPOSER(parallel_transpose_v1, "naive parallel"),
    INIT_TRANSPOSER(parallel_transpose_v2, "32x32 shared mem"),
    INIT_TRANSPOSER(parallel_transpose_v3, "16x16 shared mem"),
    INIT_TRANSPOSER(parallel_transpose_v4, "16x16 tiled"),
    // INIT_TRANSPOSER(parallel_transpose_v5, "p_v5"),
  };

  printf("N\\algo ");
  for (Tranposer<ull> tranposer : tranposing_algos)
    printf("%23s ", tranposer.name.c_str());
  printf("\n");

  for (size_t N : Ns) {
    size_t matrixSize = N * N * sizeof(ull);
    ull *h_mat = (ull *) malloc(matrixSize);
    ull *h_out = (ull *) malloc(matrixSize);
    float transposing_time;

    // make initial array
    for (size_t i = 0; i < N * N; ++i)
      h_mat[i] = rand() % N; 

    printf("%6lu ", N);

    // run each tranpose algo
    for (Tranposer<ull> tranposer : tranposing_algos) {
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
