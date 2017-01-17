#include <iostream>
#include "timer.h"
#include "utils.h"

template <typename T>
__global__
void transpose_shared_mem(const T * const d_mat, T * const d_out, const size_t N) 
{
  // extern __shared__ T s_mat[];
  extern __shared__ __align__(sizeof(T)) unsigned char s_mem[];
  T *s_mat = reinterpret_cast<T *>(s_mem);

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= N || y >= N)
    return;

  s_mat[threadIdx.x + blockDim.x * threadIdx.y] = d_mat[x + y * N];
  d_out[y + x * N] = s_mat[threadIdx.x + blockDim.x * threadIdx.y];
}

template <typename T>
float parallel_transpose_shared_mem(const T *h_matrix, T *h_output, const size_t N, const size_t K)
{
  T *d_matrix, *d_output;
  const size_t matrixSize = N * N * sizeof(T);
  const dim3 blockSize(K,K,1);
  const dim3 gridSize(round_up(N, (size_t)blockSize.x), round_up(N, (size_t)blockSize.y), 1);

  cudaMalloc((void **) &d_matrix, matrixSize);
  cudaMalloc((void **) &d_output, matrixSize);
CHECK_ERR
  cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice);

  GpuTimer timer;
  timer.Start();
  transpose_shared_mem<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(T)>>>(d_matrix, d_output, N);
CHECK_ERR
  timer.Stop();
  cudaMemcpy(h_output, d_output, matrixSize, cudaMemcpyDeviceToHost);

  cudaFree(d_matrix);
  cudaFree(d_output);
CHECK_ERR

  return timer.Elapsed();
}

template <typename T>
float parallel_transpose_v2(const T *h_matrix, T *h_output, const size_t N)
{
  return parallel_transpose_shared_mem(h_matrix, h_output, N, 32);
}

template <typename T>
float parallel_transpose_v3(const T *h_matrix, T *h_output, const size_t N)
{
  return parallel_transpose_shared_mem(h_matrix, h_output, N, 16);
}

// initialize transpose for certain types
INIT_TRANSPOSE_FUNC(parallel_transpose_v2, ui);
INIT_TRANSPOSE_FUNC(parallel_transpose_v2, ull);
INIT_TRANSPOSE_FUNC(parallel_transpose_v2, float);

INIT_TRANSPOSE_FUNC(parallel_transpose_v3, ui);
INIT_TRANSPOSE_FUNC(parallel_transpose_v3, ull);
INIT_TRANSPOSE_FUNC(parallel_transpose_v3, float);