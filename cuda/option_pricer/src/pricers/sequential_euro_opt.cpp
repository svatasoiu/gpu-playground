#include "pricers/sequential_euro_opt.h"
#include "options.h"
#include "timer.h"
#include "utils.h"

#include "utils/option_payoffs.h"
#include "utils/sequential_monte_carlo.h"
#include "utils/stats.h"

using namespace options;

namespace pricers {

template <typename T>
pricing_output<T> SimpleSequentialPricer<T>::price(pricing_args<T>& args) {
  // parse arguments
  // option_type_t opt_type = args.options->type; 
	// bool is_call = args.option->is_call;
	// T S0  = args.option->S0;
	// T K   = args.option->K;
	// T r   = args.option->r;
	// T ttm = args.option->ttm;
	// T vol = args.option->vol;
	size_t num_trials  = args.n_trials;
	// size_t path_length = args.path_len;

	// // pointer to the payoff function (i.e. put or call)
	// double (*payoff_func)(double, double) = is_call ? &euro_call_option_payoff : &euro_put_option_payoff;
	std::vector<T> estimates(num_trials);
  return {
    stats::mean(estimates),
    stats::standard_error(estimates)
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