#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "stat.h"
#include "monte_sim.h"
#include "no_reduction.h"
#include "antithetic.h"
#include "control_variate.h"

#define USEAGE """USEAGE: ./price <0 for no red | 1 for anti | 2 for CV> <'c' for call option | 'p' for put option> <S0> <K> <r> <T> <vol> <# trials> <path length> (# of threads)"""

typedef enum {
	NO_REDUCTION    = 0, 
	ANTITHETIC      = 1,
	CONTROL_VARIATE = 2
} var_red_opt_t;

void *simulate_option_wrapper(void *args);
static inline double call_option_payoff(double, double);
static inline double put_option_payoff(double, double);

int main(int argc, char **argv) {
	if (argc < 10 || argc > 11) {
		printf("%s\n", USEAGE);
		exit(1);
	} 

	// parse arguments
	var_red_opt_t var_reduct_opt = (var_red_opt_t) atoi(argv[1]);
	int is_call = argv[2][0];
	double S0  = strtod(argv[3], NULL);
	double K   = strtod(argv[4], NULL);
	double r   = strtod(argv[5], NULL);
	double T   = strtod(argv[6], NULL);
	double vol = strtod(argv[7], NULL);
	int num_trials  = atoi(argv[8]); 
	int path_length = atoi(argv[9]);

	int (*mc_func)(monte_args_t *) = NULL;
	switch (var_reduct_opt) {
		case NO_REDUCTION:    mc_func = &mc_no_reduction; break;
		case ANTITHETIC:      mc_func = &mc_antithetic; break;
		case CONTROL_VARIATE: mc_func = &mc_control_variate; break;
		default:
			printf("The only options for variance reduction are 0=NO_REDUCTION, 1=ANTITHETIC, or 2=CONTROL_VARIATE\n");
			exit(1);
	}

	if (!(is_call == 'c' || is_call == 'p')) {
		printf("ERROR: is_call should either be 'c' or 'p'\n");
		printf("%s\n", USEAGE);
		exit(1);
	}

	// round down to nearest even number
	// since Box-Muller generates N(0,1) samples 2 at a time,
	// we simulate paths in pairs.
	// I know this gets rid of one sample if num_trials is odd,
	// but it shouldn't be a big issue if num_trials is large.
	num_trials = num_trials - (num_trials % 2); 

	// pointer to the payoff function (i.e. put or call)
	double (*payoff_func)(double, double) = is_call == 'c' ? &call_option_payoff : &put_option_payoff;

	// error check
	if (S0 < 0 || K < 0 || T < 0 || vol < 0 || num_trials < 0 || path_length < 0) {
		printf("ERROR: All numeric arguments should be nonnegative (some strictly positive)\n");
		printf("%s\n", USEAGE);
		exit(1);
	}

	int num_threads = 1;
	int per_thread = num_trials;

	if (argc == 11) {
		num_threads = atoi(argv[10]);
		if (num_threads <= 0) {
			printf("ERROR: num_threads should be strictly positive\n");
			printf("%s\n", USEAGE);
			exit(1);
		}
		per_thread = num_trials / num_threads; // number of samples per thread
	}

	// crucial that we malloc, as opposed to defining the array 
	// locally (i.e. on stack), so we don't blow up our stack
	double *samples = calloc(num_trials,sizeof(double));

	// allocate space for antithetic samples if necessary
	double *antithetic_samples = NULL;
	if (var_reduct_opt == ANTITHETIC) 
		antithetic_samples = calloc(num_trials,sizeof(double));

	// do the actual simulation
	int err;
	if (num_threads == 1) {
		// no need to multithread
		monte_args_t args = { S0, r, T, K, vol, num_trials, path_length, 
			samples, antithetic_samples, time(NULL), mc_func };
		mc_func(&args);
	} else {
		pthread_t threads[num_threads];     // array of thread handles
		monte_args_t all_args[num_threads]; // array of argument structs
		memset(threads,  0, num_threads*sizeof(pthread_t));
		memset(all_args, 0, num_threads*sizeof(monte_args_t));

		for (int i = 0; i < num_threads; ++i) {
			// copy data into all_args, so that each thread can access its own data
			// thread i will compute samples with indices in the range [i * per_thread, (i+1) * per_thread)
        	monte_args_t *data = &(all_args[i]);
			data->S0 = S0; 
			data->r  = r; 
			data->T  = T; 
			data->K  = K; 
			data->vol = vol; 
			data->n_trials = per_thread;
			data->path_len = path_length; 
			data->samples = samples + i * per_thread; // start of this thread's region to put samples
			data->antithetic = antithetic_samples + i * per_thread;
			data->seed = time(NULL) + i; // create a different seed for each thread
			data->mc_func = mc_func;

			// create thread that will do these simulations
			if ((err = pthread_create(&threads[i], NULL, &simulate_option_wrapper, (void *) data))) {
				// error happened in creation
				printf("Error happened while creating thread %d: %s\n", i, strerror(err));
				exit(1);
			}
		}

		// wait for all of the threads to finish
		for (int i = 0; i < num_threads; ++i) {
			if ((err = pthread_join(threads[i], NULL))) {
				// error happened in joining
				printf("Error happened while trying to join with thread %d: %s\n", i, strerror(err));
				exit(1);
			}
		}
	}

	// now we have the samples. need to compute estimates
	double *estimates = calloc(num_trials,sizeof(double));
	const double discount = exp(-r * T);

	// compute estimates from samples now
	if (var_reduct_opt == NO_REDUCTION) {
		// compute vanilla MC estimate
		for (int i = 0; i < num_trials; ++i) {
			estimates[i] = discount * payoff_func(samples[i], K);
		}
	} else if (var_reduct_opt == ANTITHETIC) {
		// compute antithetic estimate
		for (int i = 0; i < num_trials; ++i) {
			estimates[i] = 0.5 * discount * (payoff_func(samples[i], K) + payoff_func(antithetic_samples[i], K));
		}
	} else if (var_reduct_opt == CONTROL_VARIATE) {
		// compute control variate estimate
		// again, important to (m/c)alloc here as to not blow the stack
		double *prices   = calloc(num_trials,sizeof(double));
		double *controls = calloc(num_trials,sizeof(double));

		// the control I use is simply the discounted stock price
		// since it follows GBM, its expected value is S0*e^(rT)
		// thus E[discount * S_T] = S0, a known value
		for (int i = 0; i < num_trials; ++i) {
			prices[i]   = discount * payoff_func(samples[i], K);
			controls[i] = discount * samples[i] - S0;
		}

		// compute optimal b
		double price_avg = mean(prices, num_trials);
		double control_avg = mean(controls, num_trials);
		double numerator = 0., denominator = 0.;
		for (int i = 0; i < num_trials; ++i) {
			numerator   += (prices[i] - price_avg) * (controls[i] - control_avg);
			denominator += (controls[i] - control_avg) * (controls[i] - control_avg);
		}
		double b = numerator/denominator;
		printf("optimal b for control variate reduction: %f\n", b);

		// compute estimates using b
		for (int i = 0; i < num_trials; ++i) {
			estimates[i] = prices[i] - b * controls[i];
		}
		free(controls);
		free(prices);
	}

	free(samples);
	if (var_reduct_opt == ANTITHETIC) 
		free(antithetic_samples);

	// print mean and standard error of estimate
	printf("MEAN: %f, SE: %f\n", 
		mean(estimates, num_trials), 
		standard_error(estimates, num_trials));

	free(estimates);
	return 0;
}

// wrapper for simulate_option that has the signature required by pthread_create
void *simulate_option_wrapper(void *args) {
	monte_args_t *mc_args = (monte_args_t *) args;
	mc_args->mc_func(mc_args);
	return NULL;
}

// Option pay off functions 
static inline double call_option_payoff(double S, double K) {
	return MAX(S - K, 0);
}

static inline double put_option_payoff(double S, double K) {
	return MAX(K - S, 0);
}