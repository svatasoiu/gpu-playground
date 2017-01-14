#include <cassert>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "timer.h"
#include "utils.h"

#define MAKE_SCANNER(f) { f, #f }

typedef float (*scan_func_t)(const ull * const, ull * const, int);

struct scanner_t {
  scan_func_t scan;
  std::string name;
};

float hillis_steele_scan(const ull * const, ull * const, int);
float blocked_scan(const ull * const, ull * const, int);
float blelloch_scan(const ull * const, ull * const, int);
float shared_blelloch_scan(const ull * const, ull * const, int);

int main(int argc, char **argv) {
  if (argc < 2) {
		std::cout << "USAGE: " << argv[0] << " <# to square>" << std::endl;
		exit(1);
	}

  int N = atoi(argv[1]);
  size_t arraySize = N * sizeof(ull);
  ull *h_arr = (ull *) malloc(arraySize);
  ull *h_out = (ull *) malloc(arraySize);
  ull *h_gpu_out = (ull *) malloc(arraySize);
  ull cpu_time;
  float gpu_time;

  scanner_t parallel_algos[] = {
    MAKE_SCANNER(hillis_steele_scan), 
    MAKE_SCANNER(blocked_scan),
    // MAKE_SCANNER(blelloch_scan),
    // MAKE_SCANNER(shared_blelloch_scan)
  };

  // make initial array
  for (int i = 0; i < N; ++i)
    h_arr[i] = i; // TODO: check this

  // time naive, sequential method
  timeval tv;
  gettimeofday(&tv, 0);
  cpu_time = tv.tv_usec;
  ull sum = 0;
  for (int i = 0; i < N; ++i) {
    sum += h_arr[i];
    h_out[i] = sum;
  }
  gettimeofday(&tv, 0);
  cpu_time = tv.tv_usec - cpu_time;

  printf("Sequential code code ran in: %.2f ms.\n", (float)cpu_time/1000.f);

  // set up gpu
  for (scanner_t scanner : parallel_algos) {
    gpu_time = scanner.scan(h_arr, h_gpu_out, N);
    cudaDeviceSynchronize(); 

    printf("Algo %s took %.2f ms\n", scanner.name.c_str(), gpu_time);

    // check h_gpu_out matches h_out
    for (int i = 0; i < N; ++i) {
      if (h_out[i] != h_gpu_out[i]) {
        std::cout << i << " " << h_arr[i] << " " << h_out[i] << " " << h_gpu_out[i] << std::endl;
        assert(false && "gpu output didn't match");
      } 
    }
  }

  free(h_arr);

  return 0;
}
