#pragma once

#include <ostream>
#include <stdio.h>
#include <string>

namespace options {

enum var_red_t {
	NO_REDUCTION    = 0, 
	ANTITHETIC      = 1,
	CONTROL_VARIATE = 2
};

enum option_type_t {
	EUROPEAN,
	AMERICAN,
	ASIAN
};

// T is typically float or double
template <typename T>
struct option_params {
	T S0;  // initial stock price
	T r;   // risk-free interest rate
	T ttm;   // time to maturity
	T K;   // strike price
	T vol; // volatility

	option_type_t type;
	bool is_call; // true for call, false for put (if applicable)
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const option_params<T>& o) {
	return os << o.type << " " 
			  << o.is_call << " " 
			  << o.S0 << " " 
			  << o.r << " " 
			  << o.ttm << " " 
			  << o.K << " " 
			  << o.vol;
}

}

namespace pricers {

template <typename T>
struct pricing_args {
	options::option_params<T> option;
	size_t    n_trials; // number of trials to simulate
	size_t    path_len; // number of steps in path from 0->T
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const pricing_args<T>& pargs) {
	return os << pargs.option << " " << pargs.n_trials << " " << pargs.path_len;
}

template <typename T>
pricing_args<T> parse_args(const std::string& line) {
	return {0, 0, 0};
}

template <typename T>
struct pricing_output {
	T price;
	T variance;
	
	float pricing_time;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const pricing_output<T>& pout) {
	return os << "est: " << pout.price 
			  << ", var: " << pout.variance
			  << ", time: " << pout.pricing_time;
}

template <typename T>
class Pricer {
public:
	virtual ~Pricer() { ; };
	virtual pricing_output<T> price(pricing_args<T>&) = 0;
	virtual std::string getName() = 0;
};

}