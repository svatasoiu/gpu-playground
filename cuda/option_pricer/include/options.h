#pragma once

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
	option_type_t type;

	bool is_call; // true for call, false for put (if applicable)
	T S0;  // initial stock price
	T r;   // risk-free interest rate
	T ttm;   // time to maturity
	T K;   // strike price
	T vol; // volatility
};

template <typename T>
struct pricing_args {
	option_params<T> *option;
	size_t    n_trials; // number of trials to simulate
	size_t    path_len; // number of steps in path from 0->T

	void display() {
		printf("%lu %lu\n", n_trials, path_len);
	}
};

template <typename T>
pricing_args<T> parse_args(const std::string& line) {
	return {NULL, 0, 0};
}

template <typename T>
struct pricing_output {
	T price;
	T variance;
	
	float pricing_time;

	void display() {
		printf("%f %f %f\n", price, variance, pricing_time);
	}
};

template <typename T>
class Pricer {
public:
	virtual ~Pricer() { ; };
	virtual pricing_output<T> price(pricing_args<T>&) = 0;
};

}
