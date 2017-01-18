#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "stat.h"
#include "monte_sim.h"
#include "no_reduction.h"

#define USEAGE """USEAGE: ./baseline_price <'c' for call option | 'p' for put option> <S0> <K> <r> <T> <vol> <# trials> <path length>"""

static inline double call_option_payoff(double, double);
static inline double put_option_payoff(double, double);

int main(int argc, char **argv) {
	if (argc != 9) {
		printf("%s\n", USEAGE);
		exit(1);
	} 

	// parse arguments
	int is_call = argv[1][0];
	double S0  = strtod(argv[2], NULL);
	double K   = strtod(argv[3], NULL);
	double r   = strtod(argv[4], NULL);
	double T   = strtod(argv[5], NULL);
	double vol = strtod(argv[6], NULL);
	int num_trials  = atoi(argv[7]);
	int path_length = atoi(argv[8]);

	if (!(is_call == 'c' || is_call == 'p')) {
		printf("ERROR: is_call should either be 'c' or 'p'\n");
		printf("%s\n", USEAGE);
		exit(1);
	}

	// pointer to the payoff function (i.e. put or call)
	double (*payoff_func)(double, double) = is_call == 'c' ? &call_option_payoff : &put_option_payoff;

	// error check
	if (S0 < 0 || K < 0 || T < 0 || vol < 0 || num_trials < 0 || path_length < 0) {
		printf("ERROR: All numeric arguments should be nonnegative (some strictly positive)\n");
		printf("%s\n", USEAGE);
		exit(1);
	}
	
	// round down to nearest even number
	// since Box-Muller generates N(0,1) samples 2 at a time,
	// we simulate paths in pairs.
	// I know this gets rid of one sample if num_trials is odd,
	// but it shouldn't be a big issue if num_trials is large.
	num_trials = num_trials - (num_trials % 2); 

	// crucial that we malloc, as opposed to defining the array 
	// locally (i.e. on stack), so we don't blow up our stack
	double *samples = calloc(num_trials,sizeof(double));

	// no need to multithread
	monte_args_t args = { S0, r, T, K, vol, num_trials, path_length, 
		samples, NULL, time(NULL), &mc_no_reduction };
	mc_no_reduction(&args);

	// now we have the samples. need to compute estimates
	double *estimates = calloc(num_trials,sizeof(double));
	const double discount = exp(-r * T);

	// compute estimates from samples now
	// compute vanilla MC estimate
	for (int i = 0; i < num_trials; ++i)
		estimates[i] = discount * payoff_func(samples[i], K);

	free(samples);

	// print mean and standard error of estimate
	printf("MEAN: %f, SE: %f\n", 
		mean(estimates, num_trials), 
		standard_error(estimates, num_trials));

	free(estimates);
	return 0;
}

// Option pay off functions 
static inline double call_option_payoff(double S, double K) {
	return MAX(S - K, 0);
}

static inline double put_option_payoff(double S, double K) {
	return MAX(K - S, 0);
}