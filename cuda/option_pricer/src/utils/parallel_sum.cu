#include "utils.h"

template <typename T>
__global__
void sum_kernel(const T * const d_arr, T * const d_out, int N)
{
  extern __shared__ __align__(sizeof(T)) unsigned char s_mem[];
  T *s_arr = reinterpret_cast<T *>(s_mem);

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

namespace parallel {

template <typename T>
T sum(const T * const d_arr, const size_t N)
{
  const size_t blockSize = 1024; 
  size_t gridSize = round_up(N, blockSize); 
  
  T *d_copy, *d_out;
  size_t arraySize = N * sizeof(T);
  checkCudaErrors(cudaMalloc((void**)&d_copy, arraySize));
  checkCudaErrors(cudaMalloc((void**)&d_out, gridSize * sizeof(T)));

  cudaMemcpy(d_copy, d_arr, arraySize, cudaMemcpyDeviceToDevice);

  size_t numElts = N;
  while (numElts >= blockSize) {
    sum_kernel<<<gridSize, blockSize, blockSize * sizeof(T)>>>(d_copy, d_out, numElts);

    // swap
    T *tmp = d_copy;
    d_copy = d_out;
    d_out = tmp;

    numElts = gridSize;
    gridSize = round_up(numElts,blockSize);
  }

  T *h_out = (T *) malloc(numElts * sizeof(T));
  checkCudaErrors(cudaMemcpy(h_out, d_copy, numElts * sizeof(T), cudaMemcpyDeviceToHost));

  T s = 0;
  for (size_t i = 0; i < numElts; ++i)
    s += h_out[i];
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  cudaFree(d_copy);
  cudaFree(d_out);
  free(h_out);

  return s;
}


// initialize sum for certain types
#define INIT_SUM(f, T) template T f<T>(const T * const, const size_t);
INIT_SUM(sum, float);
INIT_SUM(sum, double);

}