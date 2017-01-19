#include <string>

#include "pricers/black_scholes_euro_opt.h"
#include "options.h"
#include "timer.h"
#include "utils.h"

using namespace options;

namespace pricers {

template <typename T>
pricing_output<T> BlackScholesEuroPricer<T>::price(pricing_args<T>&) {
  return {0,0,0.5f};
}

template <typename T>
std::string BlackScholesEuroPricer<T>::getName() {
  return "BlackScholesEuroPricer";
}

// initialize transpose for certain types
INIT_PRICER(BlackScholesEuroPricer, float);
INIT_PRICER(BlackScholesEuroPricer, double);

}