#include <cstring>
#include "utils.h"

template <class T>
void swap_ptr(T *p1, T *p2) {
	T tmp = *p1;
	*p1 = *p2;
	*p2 = tmp;
}

template <class T>
int partition(T *arr, int len, int partitionIndex) {
	int store = 0;
	T *left = arr;
	T *right = arr + (len - 1);
	int partition = arr[partitionIndex];
	swap_ptr(arr + partitionIndex, right);

	// swap so left half is less than partition val
	for (; left < right; ++left) {
		if (*left < partition) {
			swap_ptr(left, arr + store);
			++store;
		}
	}
	swap_ptr(arr + store, right);
	return store;
}

template <class T>
void quicksort(T *arr, int len) {
	if (len <= 1) 
		return;

	int partitionIndex = 0;
	int pIndex = partition(arr, len, partitionIndex);
	quicksort(arr, pIndex); // qsort left half
	quicksort(arr + pIndex + 1, len - pIndex - 1); // sort right half
}

template <class T>
void sequential_quicksort(const T *h_input, T *h_output, const size_t N) {
  std::memcpy(h_output, h_input, N * sizeof(T));
  quicksort(h_output, N);
}

// initialize radix_sort for certain types
INIT_SORT_FUNC(sequential_quicksort, ull);