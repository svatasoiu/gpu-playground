#include "utils.h"

__global__
void square_kernel(const float * const d_in, float * const d_out, int N)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= N)
    return;

  float tmp = d_in[x];
  d_out[x] = tmp * tmp;
}

void parallel_square(const float * const d_in, float * const d_out, int N)
{
  const dim3 blockSize(256, 1, 1); 
  const dim3 gridSize( (N+blockSize.x-1)/blockSize.x, 1, 1); 
  square_kernel<<<gridSize, blockSize>>>(d_in, d_out, N);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
