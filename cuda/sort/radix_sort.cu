//Radix Sorting

#include <iostream>
#include "hillis_steele_scan.cuh"
#include "timer.h"
#include "utils.h"

template <typename T>
__global__
void count_zero_kernel(const T * const d_arr, ui * const d_out, const size_t N, const size_t binMask)
{
  extern __shared__ ui s_arr[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  if (x >= N) {
    s_arr[tid] = 0;
    return;
  }

  s_arr[tid] = (((d_arr[x] & binMask) == 0) ? 1 : 0);
  __syncthreads();

  for (ui s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) {
      s_arr[tid] += s_arr[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = s_arr[0];
  }
}

template <typename T>
__global__
void compute_ones_kernel(const T* const d_input, T* const d_newPos,
                         const ui numZero, const size_t N, const size_t binMask) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= N)
    return;

  if (d_input[x] & binMask)
    d_newPos[x] = x - d_newPos[x] + numZero;
  else
    d_newPos[x]--;
}

template <typename T>
__global__
void reorder_kernel(const T* const d_input, T* const d_output,  
                    const T* const d_newPos, const size_t N) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= N)
    return;

  d_output[d_newPos[x]] = d_input[x];
}

template <typename T>
void radix_sort(const T *h_input, T *h_output, const size_t N)
{
  // static_assert(std::is_integral<T>::value, "T must have integral value");

  T *d_input, *d_output;
  cudaMalloc((void **) &d_input, N * sizeof(T));
  cudaMalloc((void **) &d_output, N * sizeof(T));

  cudaMemcpy(d_input, h_input, N * sizeof(T), cudaMemcpyHostToDevice);

  const size_t numBits = 1;
  const size_t numBins = 1 << numBits;
  const size_t blockSize = 1024;
  const size_t gridSize = round_up(N, blockSize);

  ui numZero = 0; 
  T *d_newPos, *d_block_sums;
  cudaMalloc((void **) &d_newPos, N * sizeof(T) + gridSize * sizeof(T));
  cudaMalloc((void **) &d_block_sums, gridSize * sizeof(T));

  ui *d_block_zeros;  
  cudaMalloc((void **) &d_block_zeros, gridSize * sizeof(ui));
  ui *h_block_zeros = (ui *) malloc(gridSize * sizeof(ui));

CHECK_ERR
  for (ui i = 0; i < 8 * sizeof(ui); i += numBits) {
    size_t binMask = (numBins - 1) << i;

    count_zero_kernel<<<gridSize, blockSize, blockSize * sizeof(ui)>>>(d_input, d_block_zeros, N, binMask);
    checkCudaErrors(cudaMemcpy(h_block_zeros, d_block_zeros, gridSize * sizeof(ui), cudaMemcpyDeviceToHost));
    numZero = 0;
    for (size_t i = 0; i < gridSize; ++i) {
      numZero += h_block_zeros[i];
    }

CHECK_ERR
    hillis_steele_scan(d_input, d_newPos, d_block_sums, N, binMask, gridSize, blockSize);
    compute_ones_kernel<<<gridSize, blockSize>>>(d_input, d_newPos, numZero, N, binMask);
CHECK_ERR
    reorder_kernel<<<gridSize, blockSize>>>(d_input, d_output, d_newPos, N);

CHECK_ERR
    //swap the buffers (pointers only)
    std::swap(d_input, d_output);
  }

  std::swap(d_input, d_output);

  cudaMemcpy(h_output, d_output, N * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_block_zeros);
  cudaFree(d_block_sums);
  cudaFree(d_newPos);
  cudaFree(d_input);
  cudaFree(d_output);

  free(h_block_zeros);
}

// initialize radix_sort for certain types
INIT_SORT_FUNC(radix_sort, ull);