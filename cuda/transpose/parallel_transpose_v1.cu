//Radix Sorting

#include <iostream>
#include "timer.h"
#include "utils.h"

template <typename T>
__global__
void transpose_v1(const T * const d_mat, T * const d_out, const size_t N) 
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= N*N)
    return;

  int row = x / N;
  int col = x % N;
  d_out[row + col * N] = d_mat[col + row * N];
}

template <typename T>
float parallel_transpose_v1(const T *h_matrix, T *h_output, const size_t N)
{
  T *d_matrix, *d_output;
  const size_t matrixSize = N * N * sizeof(T);
  const size_t blockSize = 1024;
  const size_t gridSize = round_up(N * N, blockSize);

  cudaMalloc((void **) &d_matrix, matrixSize);
  cudaMalloc((void **) &d_output, matrixSize);

  cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice);

  GpuTimer timer;
  timer.Start();
  transpose_v1<<<gridSize, blockSize>>>(d_matrix, d_output, N);
  timer.Stop();
  cudaMemcpy(h_output, d_output, matrixSize, cudaMemcpyDeviceToHost);

  cudaFree(d_matrix);
  cudaFree(d_output);

  return timer.Elapsed();
}

// initialize transpose for certain types
INIT_TRANSPOSE_FUNC(parallel_transpose_v1, ull);