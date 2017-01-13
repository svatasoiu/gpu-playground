#include "utils.h"
#include "timer.h"
#include <stdio.h>

__global__
void sum_kernel(ull * const d_arr, ull * const d_out, int N)
{
  extern __shared__ ull s_arr[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  if (x >= N)
    return;

  s_arr[tid] = d_arr[x];
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) {
      s_arr[tid] += s_arr[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = s_arr[0];
  }
}

ull parallel_sum(const ull * const h_arr, int N)
{
  const size_t blockSize = 1024; 
  size_t gridSize  = (N+blockSize-1)/blockSize; 
  
  ull *d_arr, *d_out;
  size_t arraySize = N * sizeof(ull);
  checkCudaErrors(cudaMalloc((void**)&d_arr, arraySize));
  checkCudaErrors(cudaMalloc((void**)&d_out, gridSize * sizeof(ull)));

  // host -> device
  checkCudaErrors(cudaMemcpy(d_arr, h_arr, arraySize, cudaMemcpyHostToDevice));
  

  GpuTimer timer;
  timer.Start();
  size_t numElts = N;
  while (numElts >= blockSize) {
    sum_kernel<<<gridSize, blockSize, blockSize * sizeof(ull)>>>(d_arr, d_out, numElts);

    // swap
    ull *tmp = d_arr;
    d_arr = d_out;
    d_out = tmp;

    numElts = gridSize;
    gridSize = (numElts+blockSize-1)/blockSize;
  }
  timer.Stop();
  cudaDeviceSynchronize();
  ull *h_out = (ull *) malloc(numElts * sizeof(ull));
  checkCudaErrors(cudaMemcpy(h_out, d_arr, numElts * sizeof(ull), cudaMemcpyDeviceToHost));

  ull sum = 0;
  for (size_t i = 0; i < numElts; ++i)
    sum += h_out[i];
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  cudaFree(d_arr);
  cudaFree(d_out);
  free(h_out);

printf("Your parallel code ran in: %f msecs.\n", timer.Elapsed());

  return sum;
}
