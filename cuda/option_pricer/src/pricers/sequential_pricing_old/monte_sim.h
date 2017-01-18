#pragma once

#define MAX(x, y) ((x) > (y)? (x) : (y))

typedef struct monte_args {
	double S0;  // initial stock price
	double r;   // risk-free interest rate
	double T;   // time to maturity
	double K;   // strike price
	double vol; // volatility
	int    n_trials; // number of trials to simulate
	int    path_len; // number of steps in path from 0->T

	double *samples;    // output array for estimates produced by these simulations
	double *antithetic; // output array for secondary estimates (only used in antithetic variance reduction)
	unsigned int seed;  // seed for thread doing these simulations
	int (*mc_func)(struct monte_args *); // function to use for sampling
} monte_args_t;
