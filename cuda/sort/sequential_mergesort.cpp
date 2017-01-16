#include <cstring>
#include "utils.h"

template <class T>
void merge(T *arr, int begin, int mid, int end, T *dst) {
	int iLeft = begin; 
	int iRight = mid;

	// could create local working arrays here instead of having dst.

	for (int i = begin; i < end; ++i) {
		if (iLeft < mid && (iRight >= end || arr[iLeft] < arr[iRight])) {
			dst[i] = arr[iLeft++];
		} else {
			dst[i] = arr[iRight++];
		}
	}
}

template <class T>
void mergesort(T *arr, int begin, int end, T *dest) {
	if (end - begin <= 1) {
		return;
	}

	int mid = (begin + end) / 2;
	mergesort(dest, begin, mid, arr);
	mergesort(dest, mid, end, arr);
	merge(arr, begin, mid, end, dest);
}

template <class T>
void sequential_mergesort(const T *h_input, T *h_output, const size_t N) {
  T *tmp = (T *) malloc(N * sizeof(T));
  std::memcpy(tmp, h_input, N * sizeof(T));
  mergesort(tmp, 0, N, h_output);
}

// initialize radix_sort for certain types
INIT_SORT_FUNC(sequential_mergesort, ull);