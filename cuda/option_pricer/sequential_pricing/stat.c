#include <math.h>

// compute the average of the first len elements of arr
double mean(double arr[], int len) {
	double sum = 0.; 
	for (int i = 0; i < len; ++i)
		sum += arr[i];
	return sum/len;
}

// computes the standard error of the first len elements of arr
double standard_error(double arr[], int len) {
	double tmp = 0.;
	for (int i = 0; i < len; ++i) 
		tmp += arr[i]*arr[i];
	double avg = mean(arr, len);
	tmp -= len*avg*avg;
	return sqrt(tmp/len/(len-1));
}