#include "pricers/parallel_euro_opt.h"
#include "options.h"
#include "timer.h"
#include "utils.h"
#include "utils/parallel_primitives.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

using namespace options;

#define MAX(x,y) ((x)>(y)?(x):(y))

// cuda kernels
template <typename T>
__global__
void simulate_euro_payoff_kernel(T * const d_payoffs, const ui seed, const bool is_call,
        const T S0, const T t, const T r, const T vol, const T K, const size_t N) 
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= N)
    return;

  curandState_t state;
  curand_init(seed + N * x, 0, 0, &state);
  const T rnd = (T) curand_normal(&state);
	const double c1 = (r - vol * vol / 2.0) * t;
	const double c2 = vol * sqrt(t);
  const double S = exp(log(S0) + c1 + c2 * rnd);

  d_payoffs[x] = (T) (exp(-r*t) * (is_call ? MAX(S-K,0) : MAX(K-S,0)));
}

namespace pricers {

template <typename T>
pricing_output<T> SimpleParallelPricer<T>::price(pricing_args<T>& args) {
  option_params<T> o = args.option;
  T *d_payoffs;
  
  const size_t N = args.n_trials;
  const size_t arraySize = N * sizeof(T);
  const dim3 blockSize(1024,1,1);
  const dim3 gridSize(round_up(N, (size_t)blockSize.x), 1, 1);

  cudaMalloc((void **) &d_payoffs, arraySize);
  
CHECK_ERR
  simulate_euro_payoff_kernel<<<gridSize, blockSize>>>(d_payoffs, time(0), o.is_call, o.S0, o.ttm, o.r, o.vol, o.K, N);
CHECK_ERR
  T avg = parallel::sum(d_payoffs, N) / N;

  cudaFree(d_payoffs);
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