#include <cassert>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

DECL_TRANSPOSE_FUNC(parallel_transpose_v1);
DECL_TRANSPOSE_FUNC(sequential_transpose_v1);

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Error: need to supply at least one matrix size to benchmark\nUSAGE: %s [N]+\n", argv[0]);
		exit(1);
	}

  const size_t numBenchmarks = argc - 1; 
  size_t Ns[numBenchmarks];
  for (size_t i = 0; i < numBenchmarks; ++i)
    Ns[i] = atoi(argv[i+1]);
  
  Tranposer<ull> tranposing_algos[] = {
    INIT_TRANSPOSER(parallel_transpose_v1, "p_v1"),
    INIT_TRANSPOSER(sequential_transpose_v1, "s_v1"),
  };

  printf("N\\algo ");
  for (Tranposer<ull> tranposer : tranposing_algos)
    printf("%11s ", tranposer.name.c_str());
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
      transposing_time = tranposer.transpose(h_mat, h_out, N);
      cudaDeviceSynchronize(); 

      // TODO: color this based on correctness
      printf("%8.2f ms ", transposing_time);

      // check h_mat is indeed tranpose of h_mat
      assert(check_matrices(h_mat, h_out, N) && "output matrix is not tranpose of original");
    }

    printf("\n");

    free(h_out);
    free(h_mat);
  }

  return 0;
}
