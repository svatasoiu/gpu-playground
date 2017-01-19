#include "pricers/parallel_euro_opt.h"
#include "options.h"
#include "timer.h"
#include "utils.h"

using namespace options;

namespace pricers {

template <typename T>
pricing_output<T> SimpleParallelPricer<T>::price(pricing_args<T>&) {
  return {0,0,0.5f};
}

template <typename T>
std::string SimpleParallelPricer<T>::getName() {
  return "SimpleParallelPricer";
}

// initialize transpose for certain types
INIT_PRICER(SimpleParallelPricer, float);
INIT_PRICER(SimpleParallelPricer, double);

}