#include <iostream>
#include "timer.h"
#include "utils.h"

template <typename T>
__global__
void transpose_v5(const T * const d_mat, T * const d_out, const size_t N) 
{
  extern __shared__ __align__(sizeof(T)) unsigned char s_mem[];
  T *s_mat = reinterpret_cast<T *>(s_mem);

  int tix = threadIdx.x, tiy = threadIdx.y;
  int bdx = blockDim.x, bdy = blockDim.y;
  int x = blockIdx.x * bdx + tix;
  int y = blockIdx.y * bdy + tiy;

  if (x >= N || y >= N)
    return;

  s_mat[tiy + (bdy+1) * tix] = d_mat[x + y * N];
  __syncthreads();

  d_out[tix + blockIdx.y * bdy + (tiy + blockIdx.x * bdx) * N] = s_mat[tix + (bdx+1) * tiy];
}

template <typename T>
float parallel_transpose_v5(const T *h_matrix, T *h_output, const size_t N)
{
  T *d_matrix, *d_output;
  const size_t matrixSize = N * N * sizeof(T);
  const dim3 blockSize(16,16,1);
  const dim3 gridSize(round_up(N, (size_t)blockSize.x), round_up(N, (size_t)blockSize.y), 1);

  cudaMalloc((void **) &d_matrix, matrixSize);
  cudaMalloc((void **) &d_output, matrixSize);
CHECK_ERR
  cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice);

  GpuTimer timer;
  timer.Start();
  transpose_v5<<<gridSize, blockSize, blockSize.x * (blockSize.y + 1) * sizeof(T)>>>(d_matrix, d_output, N);
CHECK_ERR
  timer.Stop();
  cudaMemcpy(h_output, d_output, matrixSize, cudaMemcpyDeviceToHost);

  cudaFree(d_matrix);
  cudaFree(d_output);
CHECK_ERR

  return timer.Elapsed();
}

// initialize transpose for certain types
INIT_TRANSPOSE_FUNC(parallel_transpose_v5, ui);
INIT_TRANSPOSE_FUNC(parallel_transpose_v5, ull);
INIT_TRANSPOSE_FUNC(parallel_transpose_v5, float);