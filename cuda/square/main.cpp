// basic template from udacity problem set 1

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "timer.h"
#include "utils.h"

void parallel_square(const float * const h_arr, float * const d_arr, int N);

int main(int argc, char **argv) {
  if (argc < 2) {
		std::cout << "USAGE: " << argv[0] << " <# to square>" << std::endl;
		exit(1);
	}

  int N = atoi(argv[1]);
  float *h_arr = (float *) malloc(N * sizeof(float));
  float *h_seq = (float *) malloc(N * sizeof(float));
  float *h_par = (float *) malloc(N * sizeof(float));
  long long cpu_time;

  // make initial array
  for (int i = 0; i < N; ++i)
    h_arr[i] = float(i);

  // time naive, sequential method
  timeval tv;
  gettimeofday(&tv, 0);
  cpu_time = tv.tv_usec;
  for (int i = 0; i < N; ++i)
    h_seq[i] = h_arr[i] * h_arr[i];
  gettimeofday(&tv, 0);
  cpu_time = tv.tv_usec - cpu_time;

  printf("Sequential code code ran in: %lld us.\n",cpu_time);

  // set up gpu
  float *d_in, *d_out;
  checkCudaErrors(cudaMalloc((void**)&d_in, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_out, N * sizeof(float)));

  // host -> device
  checkCudaErrors(cudaMemcpy(d_in, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

  // time gpu method
  GpuTimer timer;
  timer.Start();
  parallel_square(d_in, d_out, N);
  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // host <- device
  checkCudaErrors(cudaMemcpy(h_par, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

  int err = printf("Your parallel code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  cudaFree(d_in);
  cudaFree(d_out);
  free(h_arr);
  free(h_seq);
  free(h_par);

  return 0;
}
