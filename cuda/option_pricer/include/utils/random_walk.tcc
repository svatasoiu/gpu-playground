#include <cassert>
#include <cmath>
#include <math.h>

#include "utils/sequential_monte_carlo.h"

namespace monte_carlo {

template <class T>
RandomWalkGenerator<T>::RandomWalkGenerator(ValueT S0, ValueT t, ValueT r, ValueT vol, size_t path_len) 
	: S0(S0), t(t), r(r), vol(vol), path_len(path_len), distribution(0.0, 1.0) {
	assert(S0 > 0 && "initial stock price must be positive");
	assert(t >= 0 && "ttm must be non negative");
	assert(vol >= 0 && "vol must be non negative");
	assert(path_len > 0 && "path length must be at least 1");
}

template <class T>
RandomWalkGenerator<T>::~RandomWalkGenerator() {}

template <class T>
void RandomWalkGenerator<T>::generateSample(T& path) {
	path.resize(path_len);
    ValueT log_price = log(S0);
	const ValueT dt = t / path_len; // time step
	const ValueT c1 = (r - vol * vol / 2.0) * dt;
	const ValueT c2 = vol * sqrt(dt);

	for (size_t i = 0; i < path_len; ++i) {
		path[i] = exp(log_price);
		ValueT rnd = distribution(generator);
		log_price += c1 + c2 * rnd;
	}
}

}