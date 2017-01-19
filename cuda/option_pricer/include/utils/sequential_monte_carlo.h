#pragma once

#include "utils/stats.h"
#include <vector>

// generic monte carlo simulator 
// produces samples, then combines them into estimate
namespace monte_carlo {

enum var_red_t {
	NO_REDUCTION    = 0, 
	ANTITHETIC      = 1,
	CONTROL_VARIATE = 2
};

template <class T>
class SampleGenerator {
public:
    SampleGenerator();
    virtual ~SampleGenerator() = 0;

    virtual T generateSample() = 0;
};

template <class SampleT, class OutputT>
class PayoffCalculator {
public:
    PayoffCalculator();
    virtual ~PayoffCalculator() = 0;
    
    virtual OutputT calculate(const SampleT&) = 0;
};

template <typename T>
struct MonteCarloResult {
	T estimate;
	T variance;
};

template <class SampleT, class OutputT>
class MonteCarlo {
    // static assert that outputT is numeric
protected:
    const SampleGenerator<SampleT> &generator;
    const PayoffCalculator<SampleT,OutputT> &payoff;
public:
    MonteCarlo(const SampleGenerator<SampleT> &g, const PayoffCalculator<SampleT,OutputT> &p) 
        : generator(g), payoff(p) { ; }

    virtual ~MonteCarlo() { ; }

    virtual MonteCarloResult<OutputT> estimate(size_t) = 0; // # trials
};

template <class SampleT, class OutputT>
class SimpleMonteCarlo : public MonteCarlo<SampleT, OutputT> {
public:
    SimpleMonteCarlo(const SampleGenerator<SampleT> &g, const PayoffCalculator<SampleT,OutputT> &p) 
        : MonteCarlo<SampleT, OutputT>(g, p) { ; }
    virtual ~SimpleMonteCarlo();

    virtual MonteCarloResult<OutputT> estimate(size_t num_trials) {
        std::vector<OutputT> estimates(num_trials);

        for (size_t i = 0; i < num_trials; ++i)
            estimates[i] = this->payoff.calculate(this->generator.generateSample());

        return {
            stats::mean(estimates), 
            stats::standard_error(estimates)
        };
    }
};

/*
template <class T>
class AntitheticMonteCarlo : public MonteCarlo<T> {
public:
    MonteCarlo(SampleGenerator<T>, Payoff<T>);
    virtual ~MonteCarlo();

    virtual estimate(size_t);
}
*/
}

/*
template <typename T>
struct monte_args_t {
	T S0;  // initial stock price
	T r;   // risk-free interest rate
	T ttm;   // time to maturity
	T K;   // strike price
	T vol; // volatility
	size_t    n_trials; // number of trials to simulate
	size_t    path_len; // number of steps in path from 0->T

	std::vector<T>& samples;    // output array for estimates produced by these simulations
	std::vector<T>& antithetic; // output array for secondary estimates (only used in antithetic variance reduction)
	unsigned int seed;  // seed for thread doing these simulations
	int (*mc_func)(monte_args_t&); // function to use for sampling
};

int mc_no_reduction(monte_args_t &args);
int mc_antithetic(monte_args_t &args);
int mc_control_variate(monte_args_t &args);
*/