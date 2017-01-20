#include "pricers/sequential_euro_opt.h"
#include "options.h"
#include "timer.h"
#include "utils.h"

#include "utils/option_payoffs.h"
#include "utils/sequential_monte_carlo.h"
#include "utils/stats.h"

#include <vector>

using namespace options;

namespace pricers {

template <typename T>
pricing_output<T> SimpleSequentialPricer<T>::price(pricing_args<T>& args) {
  auto o = args.option;
  monte_carlo::RandomWalkGenerator<std::vector<T>> rw(o.S0, o.ttm, o.r, o.vol, args.path_len);
  monte_carlo::EuropeanPathPayoff<std::vector<T>> epp(o.is_call, o.K);
  monte_carlo::SimpleMonteCarlo<std::vector<T>, T> mc(rw, epp);
  
  auto res = mc.estimate(args.n_trials);

  return {
    res.estimate,
    res.stderr
  };
}

template <typename T>
std::string SimpleSequentialPricer<T>::getName() {
  return "SimpleSequentialPricer";
}

// initialize transpose for certain types
INIT_PRICER(SimpleSequentialPricer, float);
INIT_PRICER(SimpleSequentialPricer, double);

}