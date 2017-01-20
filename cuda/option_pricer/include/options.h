#pragma once

#include <istream>
#include <ostream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <stdexcept>
#include <vector>

namespace options {

enum option_type_t {
	EUROPEAN,
	AMERICAN,
	ASIAN
};

static std::ostream& operator<<(std::ostream& os, const option_type_t& t) {
	switch (t) {
	case EUROPEAN:
		return os << "european";
	case AMERICAN:
		return os << "american";
	case ASIAN:
		return os << "asian";
	default:
		return os << "UNKNOWN";
	}
}

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
			  << (o.is_call ? "call" : "put" ) << " S0=$" 
			  << o.S0 << " K=$" 
			  << o.K << " Ttm=" 
			  << o.ttm << "yrs r="
			  << o.r << " vol="  
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
struct pricing_output {
	T price;
	T stderr;
	
	float pricing_time;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const pricing_output<T>& pout) {
	return os << "est: " << pout.price 
			  << ", stderr: " << pout.stderr
			  << ", time: " << pout.pricing_time;
}

template <typename T>
class Pricer {
public:
	virtual ~Pricer() { ; };
	virtual pricing_output<T> price(pricing_args<T>&) = 0;
	virtual std::string getName() = 0;
};

// this is going to be messy/hacky
template <typename T>
pricing_args<T> parse_args(const std::string& line) {
	pricing_args<T> pargs = {0};
	std::stringstream ss;
    ss.str(line);
	ss >> pargs;
	return pargs;
}

template <typename T>
std::istream& parse_euro_args(std::istream& is, pricing_args<T>& pargs) {
	std::string tmp;
	is >> tmp;
	if (tmp == "c" || tmp == "call") {
		pargs.option.is_call = true;
	} else if (tmp == "p" || tmp == "put") {
		pargs.option.is_call = false;
	} else {
		throw std::invalid_argument("must specify call or put, got: " + tmp);
	}

	return is >> pargs.option.S0 >> pargs.option.K  
			  >> pargs.option.ttm >> pargs.option.r
			  >> pargs.option.vol >> pargs.n_trials
			  >> pargs.path_len;
}

template <typename T>
std::istream& operator>>(std::istream& is, pricing_args<T>& pargs) {
	std::string tmp;
    is >> tmp;
	if (tmp == "euro" || tmp == "european") {
		pargs.option.type = options::EUROPEAN;
		return parse_euro_args(is, pargs);
	} else {
		throw std::invalid_argument("unknown option type: " + tmp);
	}
}

}