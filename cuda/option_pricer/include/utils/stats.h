// stat.h
#pragma once

#include <numeric>
#include <vector>

namespace stats {

template <class T> T mean(std::vector<T>&);
template <class T> T standard_error(std::vector<T>&);

template <class T> 
T mean(std::vector<T>& v) {
	T sum = std::accumulate(v.begin(), v.end(), 0.);
	return sum/v.size();
}

template <class T> 
T standard_error(std::vector<T> &v) {
    auto len = v.size();
	T tmp = 0.;
	for (auto elt : v) 
		tmp += elt * elt;
	T avg = mean(v);
	tmp -= len*avg*avg;
	return sqrt(tmp/len/(len-1));
}

}