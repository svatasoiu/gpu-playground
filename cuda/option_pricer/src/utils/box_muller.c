#include <math.h>
#include <stdlib.h>

// a thread-safe way implementation of the Box-Muller transformation
// outputs two IID N(0,1) samples in out1 and out2, and uses the 
// (hopefully) thread-local seed for rand_r.
void gen_normal_random(double *out1, double *out2, unsigned int *seed) {
	// generate uniform RV in [0,1]
	double u1 = (double) rand_r(seed) / RAND_MAX; 
	double u2 = (double) rand_r(seed) / RAND_MAX;

	// compute N(0,1) samples using u1,u2
	double tmp = sqrt(-2*log(u1));
	if (out1)
		*out1 = tmp*cos(2*M_PI*u2);
	if (out2)
		*out2 = tmp*sin(2*M_PI*u2);
}