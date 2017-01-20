#include <chrono>
#include <cmath>
#include <math.h>
#include <string>

#include "pricers/black_scholes_euro_opt.h"
#include "options.h"
#include "timer.h"
#include "utils.h"

using namespace options;

namespace pricers {

template <typename T>
T normcdf(T v) {
   return 0.5 * erfc(-v * M_SQRT1_2);
}

template <typename T>
pricing_output<T> BlackScholesEuroPricer<T>::price(pricing_args<T>& pargs) {
  auto o = pargs.option;
  auto start = std::chrono::high_resolution_clock::now();
  T d1 = (log(o.S0/o.K) + (o.r + o.vol*o.vol/2)*o.ttm)/o.vol/sqrt(o.ttm);
  T d2 = d1 - o.vol * sqrt(o.ttm);
  T price = o.is_call ? normcdf(d1)*o.S0-normcdf(d2)*o.K*exp(-o.r*o.ttm) 
    : normcdf(-d2)*o.K*exp(-o.r*o.ttm)-normcdf(-d1)*o.S0;
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  
  return {
    price,
    0.,
    float(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())/100.f
  };
}

template <typename T>
std::string BlackScholesEuroPricer<T>::getName() {
  return "BlackScholesEuroPricer";
}

// initialize transpose for certain types
INIT_PRICER(BlackScholesEuroPricer, float);
INIT_PRICER(BlackScholesEuroPricer, double);

}