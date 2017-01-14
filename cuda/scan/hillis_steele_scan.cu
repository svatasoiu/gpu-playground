#include "utils.h"
#include "timer.h"
#include <cassert>
#include <iostream>

// scans single blocks at a time
// optionally computes sum as well
// based on sample nvidia implementation 
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
template <class T, bool computeSum>
__global__
void hillis_steele_scan_kernel(T * const d_arr, T * const d_out, int N)
{
  extern __shared__ T temp[];

  int bdx = blockDim.x;
	int tid = threadIdx.x;
  int x = blockIdx.x * bdx + tid;
  
  if (x >= N)
    return;

  int in = 1;
  int out = 0;

  temp[tid] = d_arr[x];
  __syncthreads();

  int s = 1;
  while (s < bdx) { // used to be N
    in = out;
    out = 1 - in;
    
    temp[out * bdx + tid] = temp[in * bdx + tid] + ((tid >= s) ? temp[in * bdx + tid - s] : 0);

    __syncthreads();
    s <<= 1;
  }

  d_out[x] = temp[out * bdx + tid];

  // fill in block sums
  if (computeSum) 
    if (threadIdx.x == blockDim.x - 1)
      d_out[N + blockIdx.x] = temp[out * bdx + tid];
}

// TODO: improve performance of this
template <class T>
__global__
void add_sums_to_scan_kernel(T * const d_arr, const T * const d_block_sums, int N)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (x >= N)
    return;

  d_arr[x] += d_block_sums[blockIdx.x];
}

// for large size arrays
// TODO: add padding for things of not multiple of blockSize 
float hillis_steele_scan(const ull * const h_arr, ull * const h_out, int N)
{
  const size_t blockSize = 512; 
  const size_t gridSize  = round_up(N, (int) blockSize);

  assert(gridSize <= MAX_GRID_SIZE && "too many elements");

  ull *d_arr, *d_out, *d_block_sums;
  size_t arraySize = N * sizeof(ull);
  size_t gridSumSize = gridSize * sizeof(ull);
  checkCudaErrors(cudaMalloc((void**)&d_arr, arraySize));
  checkCudaErrors(cudaMalloc((void**)&d_out, arraySize + gridSumSize));
  checkCudaErrors(cudaMalloc((void**)&d_block_sums, gridSumSize));

  checkCudaErrors(cudaMemcpy(d_arr, h_arr, arraySize, cudaMemcpyHostToDevice));
  
  GpuTimer timer;
  timer.Start();
  // some times fails...
  hillis_steele_scan_kernel<ull, true><<<gridSize, blockSize, 2 * blockSize * sizeof(ull)>>>(d_arr, d_out, N);
  if (gridSize > 1) {
    // figure this out
    if (gridSize > 1024) {
      // block sums : 1 2 3 ... 1024 1 2 3 ... 1024
      // form block scans over these to get
      //              1 3 6 ... <..> 1 3 6 ... <..>
      const size_t gridSize2 = round_up((int) gridSize, 1024); 
      hillis_steele_scan_kernel<ull, false><<<gridSize2, 1024, 2 * 1024 * sizeof(ull)>>>(d_out + N, d_block_sums, gridSize);
      ull *h_block_sums = (ull *) malloc(gridSumSize);
      checkCudaErrors(cudaMemcpy(h_block_sums, d_block_sums, gridSumSize, cudaMemcpyDeviceToHost));
      ull prev = 0;
      for (int i = 1024; i < gridSize; ++i) {
        if (i % 1024 == 0)
          prev = h_block_sums[i-1];
        h_block_sums[i] += prev;
      }
      checkCudaErrors(cudaMemcpy(d_block_sums, h_block_sums, gridSumSize, cudaMemcpyHostToDevice));
      free(h_block_sums);
    } else {
      hillis_steele_scan_kernel<ull, false><<<1, gridSize>>>(d_out + N, d_block_sums, gridSize);
    }
    add_sums_to_scan_kernel<ull><<<gridSize-1, blockSize>>>(d_out + blockSize, d_block_sums, N - blockSize);
  }
  timer.Stop();

  checkCudaErrors(cudaMemcpy(h_out, d_out, arraySize, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(d_arr));
  checkCudaErrors(cudaFree(d_block_sums));

  return timer.Elapsed();
}