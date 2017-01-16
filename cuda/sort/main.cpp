#include <cassert>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "timer.h"
#include "utils.h"

DECL_SORT_FUNC(radix_sort);
DECL_SORT_FUNC(parallel_merge_sort);
DECL_SORT_FUNC(brick_sort);
DECL_SORT_FUNC(sequential_quicksort);
DECL_SORT_FUNC(sequential_mergesort);

int main(int argc, char **argv) {
  if (argc < 2) {
		std::cout << "USAGE: " << argv[0] << " <# to sort>" << std::endl;
		exit(1);
	}

  const size_t N = atoi(argv[1]);
  size_t arraySize = N * sizeof(ull);
  ull *h_arr = (ull *) malloc(arraySize);
  ull *h_out = (ull *) malloc(arraySize);
  float sorting_time;

  Sorter<ull> sorting_algos[] = {
    INIT_SORTER(radix_sort),
    INIT_SORTER(sequential_quicksort),
    INIT_SORTER(sequential_mergesort)
  };

  // make initial array
  for (size_t i = 0; i < N; ++i)
    h_arr[i] = rand() % N; 

  // set up gpu
  for (Sorter<ull> sorter : sorting_algos) {
    GpuTimer timer;
    timer.Start();
    sorter.sort(h_arr, h_out, N);
    timer.Stop();
    cudaDeviceSynchronize(); 

    sorting_time = timer.Elapsed();

    printf("Algo %s took %.2f ms\n", sorter.name.c_str(), sorting_time);

    // check h_gpu_out matches h_out
    for (size_t i = 1; i < N; ++i) {
      if (h_out[i] < h_out[i - 1]) {
        std::cout << sorter.name.c_str() << ": " << h_out[i-1] << " > " << h_out[i] << std::endl;
        assert(false && "output list is not sorted");
      } 
    }
  }

  free(h_out);
  free(h_arr);

  return 0;
}
