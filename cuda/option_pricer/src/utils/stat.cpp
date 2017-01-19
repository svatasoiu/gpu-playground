#include <cmath>
#include <numeric>
#include <vector>

namespace stats {

template <class T> 
T mean(std::vector<T>& v) {
	T sum = std::accumulate(v.begin(), v.end(), 0.);
	return sum/v.size();
}

template <class T> 
T standard_error(std::vector<T> &arr) {
	T tmp = 0.;
	for (size_t i = 0; i < arr.size(); ++i) 
		tmp += arr[i]*arr[i];
	T avg = mean(arr, len);
	tmp -= len*avg*avg;
	return sqrt(tmp/len/(len-1));
}

}