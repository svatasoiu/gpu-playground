#include <math.h>
#include "box_muller.h"
#include "monte_sim.h"
#include "stat.h"

// vanilla monte carlo simulation, no variance reduction
int mc_no_reduction(monte_args_t *args) {
	// bunch of constants that I don't want to recompute every loop
	// I simulated using the Log of the stock price to reduce the amount
	// of multiplications that I have to do
	const double log_price = log(args->S0);
	const double dt = args->T / args->path_len; // time step
	const double c1 = (args->r - args->vol * args->vol / 2.0) * dt;
	const double c2 = args->vol * sqrt(dt);

	// since box-muller produces two N(0,1) samples at a time
	// we just simulate in a stride of 2 as to utilize all the samples
	// also has the added benefit of 2-way loop unrolling
	for (int i = 0; i < args->n_trials; i+=2) {
		double log_S1 = log_price;
		double log_S2 = log_price;

		for (int j = 0; j < args->path_len; ++j) {
			double z1, z2;
			gen_normal_random(&z1, &z2, &args->seed);
			log_S1 += c1 + c2 * z1;
			log_S2 += c1 + c2 * z2;
		}

		args->samples[i]   = exp(log_S1);
		args->samples[i+1] = exp(log_S2);
	}
	return 0;
}