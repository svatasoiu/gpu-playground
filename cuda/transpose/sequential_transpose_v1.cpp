#include <cstring>
#include "timer.h"
#include "utils.h"

template <class T>
float sequential_transpose_v1(const T *mat, T *out, const size_t N) 
{
  GpuTimer timer;
  timer.Start();
  for (size_t i = 0; i < N; ++i)
  	for (size_t j = 0; j < N; ++j)
	  out[i + j * N] = mat[j + i * N];
  timer.Stop();

  return timer.Elapsed();
}

// initialize transpose for certain types
INIT_TRANSPOSE_FUNC(sequential_transpose_v1, ull);