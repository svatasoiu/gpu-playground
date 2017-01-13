#include "utils.h"
#include "timer.h"
#include <iostream>

using namespace std;

// scans single blocks at a time
// optionally computes sum as well
template <class T, bool computeSum>
__global__
void hillis_steele_scan_kernel(T * const d_arr, T * const d_out, int N)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
  int startOfBlock = blockIdx.x * blockDim.x;
  
  if (x >= N)
    return;

  int in = 1;
  int out = 0;

  int s = 1;
  while (s < blockDim.x) { // used to be N
    in = out;
    out = 1 - in;
    
    if (x >= startOfBlock + s) {
      d_arr[out * N + x] = d_arr[in * N + x] + d_arr[in * N + x - s];
    } else {
      d_arr[out * N + x] = d_arr[in * N + x];
    }

    __syncthreads();
    s <<= 1;
  }

  d_out[x] = d_arr[out * N + x];

  // fill in block sums
  if (computeSum) 
    if (threadIdx.x == blockDim.x - 1)
      d_out[N + blockIdx.x] = d_out[x];
}

__global__
void add_sums_to_scan_kernel(ull * const d_arr, const ull * const d_block_sums, int N)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (x >= N)
    return;

  d_arr[x] += d_block_sums[blockIdx.x];
}

// for large size arrays
// TODO: add padding for things of not size 
// TODO: think about how to handle gridSize >= blockSize
float hillis_steele_scan(const ull * const h_arr, ull * const h_out, int N)
{
  const size_t blockSize = 1024; 
  const size_t gridSize  = (N+blockSize-1)/blockSize;

  ull *d_arr, *d_out, *d_block_sums;
  size_t arraySize = N * sizeof(ull);
  size_t gridSumSize = gridSize * sizeof(ull);
  checkCudaErrors(cudaMalloc((void**)&d_arr, 2 * arraySize));
  checkCudaErrors(cudaMalloc((void**)&d_out, arraySize + 2 * gridSumSize));
  checkCudaErrors(cudaMalloc((void**)&d_block_sums, gridSumSize));

  checkCudaErrors(cudaMemcpy(d_arr, h_arr, arraySize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_arr + N, 0, arraySize));
  checkCudaErrors(cudaMemset(d_out + N + gridSize, 0, gridSumSize));
  
  GpuTimer timer;
  timer.Start();
  hillis_steele_scan_kernel<ull, true><<<gridSize, blockSize>>>(d_arr, d_out, N);
  if (gridSize > 1) {
    // works for gridSize < blockSize (hard limit of gridSize < 1024)
    hillis_steele_scan_kernel<ull, false><<<1, gridSize>>>(d_out + N, d_block_sums, gridSize);
    add_sums_to_scan_kernel<<<gridSize-1, blockSize>>>(d_out + blockSize, d_block_sums, N - blockSize);
  }
  timer.Stop();

  checkCudaErrors(cudaMemcpy(h_out, d_out, arraySize, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(d_arr));
  checkCudaErrors(cudaFree(d_block_sums));

  return timer.Elapsed();
}

__global__
void shared_hillis_steele_scan_kernel(ull * const d_arr, int N)
{
  return;
}

float shared_hillis_steele_scan(const ull * const h_arr, ull * const h_out, int N)
{
  return 0;
}