#include "pricers/parallel_euro_opt.h"
#include "options.h"
#include "timer.h"
#include "utils.h"
#include "utils/parallel_primitives.cuh"

#include <curand.h>
#include <curand_kernel.h>

using namespace options;

/* this GPU kernel function is used to initialize the random states */
__global__ void rand_init(unsigned int seed, curandState_t * const states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, x, 0, &states[x]);
}

#define MAX(x,y) ((x)>(y)?(x):(y))

// cuda kernels
template <typename T>
__global__
void simulate_euro_payoff_kernel(T * const d_payoffs, curandState_t* states, const bool is_call,
        const T S0, const T t, const T r, const T vol, const T K, const size_t N) 
{
  // extern __shared__ __align__(sizeof(T)) unsigned char s_mem[];
  // T *s_mat = reinterpret_cast<T *>(s_mem);

  const int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= N)
    return;

  const float rnd = curand_normal(&states[x]);
	const float c1 = (r - vol * vol / 2.0) * t;
	const float c2 = vol * sqrtf(t);
  const float S = expf(logf(S0) + c1 + c2 * rnd);

  d_payoffs[x] = (T) (expf(-r*t) * (is_call ? MAX(S-K,0) : MAX(K-S,0)));
}

namespace pricers {

template <typename T>
pricing_output<T> SimpleParallelPricer<T>::price(pricing_args<T>& args) {
  option_params<T> o = args.option;
  T *d_payoffs;
  curandState_t *d_states;
  
  const size_t N = args.n_trials;
  const size_t arraySize = N * sizeof(T);
  const dim3 blockSize(1024,1,1);
  const dim3 gridSize(round_up(N, (size_t)blockSize.x), 1, 1);

  cudaMalloc((void **) &d_payoffs, arraySize);
  cudaMalloc((void **) &d_states, N * sizeof(curandState_t));
  
  rand_init<<<gridSize, blockSize>>>(time(0), d_states);
CHECK_ERR
  simulate_euro_payoff_kernel<<<gridSize, blockSize>>>(d_payoffs, d_states, o.is_call, o.S0, o.ttm, o.r, o.vol, o.K, N);
CHECK_ERR

  T avg = parallel::sum(d_payoffs, N) / N;

  cudaFree(d_payoffs);
  cudaFree(d_states);
CHECK_ERR

  return {
    avg
  };
}

template <typename T>
std::string SimpleParallelPricer<T>::getName() {
  return "SimpleParallelPricer";
}

// initialize transpose for certain types
INIT_PRICER(SimpleParallelPricer, float);
INIT_PRICER(SimpleParallelPricer, double);

}