// basic template from udacity problem set 1

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "timer.h"
#include "utils.h"

ull parallel_sum(const ull * const h_arr, int N);

int main(int argc, char **argv) {
  if (argc < 2) {
		std::cout << "USAGE: " << argv[0] << " <# to square>" << std::endl;
		exit(1);
	}

  int N = atoi(argv[1]);
  int arraySize = N * sizeof(ull);
  ull *h_arr = (ull *) malloc(arraySize);
  long long cpu_time;
  ull seq_sum = 0, par_sum = 0;

  // make initial array
  for (int i = 0; i < N; ++i)
    h_arr[i] = i;

  // time naive, sequential method
  timeval tv;
  gettimeofday(&tv, 0);
  cpu_time = tv.tv_usec;
  for (int i = 0; i < N; ++i)
    seq_sum += h_arr[i];
  gettimeofday(&tv, 0);
  cpu_time = tv.tv_usec - cpu_time;

  printf("Sequential code code ran in: %lld us.\n",cpu_time);

  // set up gpu
  // time gpu method
  GpuTimer timer;
  timer.Start();
  par_sum = parallel_sum(h_arr, N);
  timer.Stop();
  cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());

  std::cout << seq_sum << " vs " << par_sum << std::endl;

  int err = printf("Parallel code+malloc: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  free(h_arr);

  return 0;
}
