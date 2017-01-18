#include "pricers/parallel_euro_opt.hpp"
#include "options.h"
#include "timer.h"
#include "utils.h"

using namespace options;

template <typename T>
pricing_output<T> SimpleParallelPricer<T>::price(pricing_args<T>&) {
  return {0,0,0.5f};
}

// initialize transpose for certain types
INIT_PRICER(SimpleParallelPricer, float);
INIT_PRICER(SimpleParallelPricer, double);