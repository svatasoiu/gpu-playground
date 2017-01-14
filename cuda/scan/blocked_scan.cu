#include "utils.h"
#include "timer.h"
#include <cassert>
#include <iostream>

// simpler kernel
// each thread scans block
template <class T>
__global__
void blocked_scan_kernel(T * const d_arr, T * const d_sums_per_block, T * const d_out, int numPerBlock, int numberOfBlocks)
{
  // this kernel sums+scans [x, x+numPerBlock)
  // places sum at d_sums_per_block[x]
  // and corresponding scan in d_out
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= numberOfBlocks)
    return;
  
  T sum = 0;
  for (int i = 0; i < numPerBlock; ++i) {
    sum += d_arr[x * numPerBlock + i];
    d_out[x * numPerBlock + i] = sum;
  }
  d_sums_per_block[x] = sum;
}

template <class T>
__global__
void blocked_add_sums_to_scan_kernel(T * const d_arr, const T * const d_block_sums, int numPerBlock, int numberOfBlocks)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= numberOfBlocks)
    return;

  for (int i = 0; i < numPerBlock; ++i) {
    d_arr[x * numPerBlock + i] += d_block_sums[x];
  }
}

struct GridParameters {
  size_t numPerBlock;
  size_t roundedN;
  size_t numberOfBlocks;
  size_t blockSize; 
  size_t gridSize;

  GridParameters(size_t N) {
    // compute numPerBlock
    blockSize = 64; //MAX_BLOCK_SIZE;
    gridSize = min((unsigned int) round_up(N, blockSize), (unsigned int) MAX_GRID_SIZE);
    numPerBlock = round_up(N, blockSize * gridSize);
    roundedN = (N%numPerBlock==0) ? N : (N+numPerBlock - (N % numPerBlock));
    numberOfBlocks = round_up(roundedN, numPerBlock);
  }

  GridParameters(size_t N, size_t _numPerBlock) {
    numPerBlock = _numPerBlock;
    roundedN = (N%numPerBlock==0) ? N : (N+numPerBlock - (N % numPerBlock));
    numberOfBlocks = round_up(roundedN, numPerBlock);
    blockSize = min((unsigned int) numberOfBlocks, (unsigned int) MAX_BLOCK_SIZE);
    gridSize = round_up(numberOfBlocks, blockSize);
  }

  void display() {
      DISPLAY(roundedN);
      DISPLAY(numPerBlock);
      DISPLAY(numberOfBlocks);
      DISPLAY(blockSize);
      DISPLAY(gridSize);
  }
};

float blocked_scan(const ull * const h_arr, ull * const h_out, const int N)
{  
  // round N to nearest multiple of blockSize
  GridParameters params(N);
//   params.display();

  assert(params.blockSize <= 1024 && "too many elements per block");
  assert(params.gridSize <= MAX_GRID_SIZE && "too many elements");

  ull *d_arr, *d_block_sums, *d_out;
  size_t arraySize = params.roundedN * sizeof(ull);
  size_t gridSumSize = params.numberOfBlocks * sizeof(ull);
  checkCudaErrors(cudaMalloc((void**)&d_arr, arraySize));
  checkCudaErrors(cudaMalloc((void**)&d_block_sums, gridSumSize));
  checkCudaErrors(cudaMalloc((void**)&d_out, arraySize));

  checkCudaErrors(cudaMemcpy(d_arr, h_arr, N * sizeof(ull), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_arr + N, 0, arraySize - (N * sizeof(ull)))); // zero excess space
  
  GpuTimer timer;
  timer.Start();
  // want this to perform as few memory accesses per thread (start max # of threads?)
  blocked_scan_kernel<ull><<<params.gridSize, params.blockSize>>>(d_arr, d_block_sums, d_out, params.numPerBlock, params.numberOfBlocks);
  // scan together d_block_sums by hand on cpu
  if (params.numberOfBlocks > 1) {
    // figure this out
    ull *h_block_sums = (ull *) malloc(gridSumSize);
    checkCudaErrors(cudaMemcpy(h_block_sums, d_block_sums, gridSumSize, cudaMemcpyDeviceToHost));
    for (int i = 1; i < params.numberOfBlocks; ++i) {
      h_block_sums[i] += h_block_sums[i-1];
    }
    checkCudaErrors(cudaMemcpy(d_block_sums, h_block_sums, gridSumSize, cudaMemcpyHostToDevice));
    free(h_block_sums);

    // want this to perform as few memory accesses per thread
    GridParameters params2(N - params.numPerBlock, params.numPerBlock);
    blocked_add_sums_to_scan_kernel<<<params2.gridSize, params2.blockSize>>>(d_out + params.numPerBlock, d_block_sums, params.numPerBlock, params2.numberOfBlocks);
  }
  timer.Stop();
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy(h_out, d_out, N * sizeof(ull), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_arr));
  checkCudaErrors(cudaFree(d_block_sums));
  checkCudaErrors(cudaFree(d_out));

  return timer.Elapsed();
}