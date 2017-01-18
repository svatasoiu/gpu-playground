#include "pricers/sequential_euro_opt.hpp"
#include "options.h"
#include "timer.h"
#include "utils.h"

using namespace options;

template <typename T>
pricing_output<T> SimpleSequentialPricer<T>::price(pricing_args<T>&) {
  return {0,0,1.f};
}

// initialize transpose for certain types
INIT_PRICER(SimpleSequentialPricer, float);
INIT_PRICER(SimpleSequentialPricer, double);