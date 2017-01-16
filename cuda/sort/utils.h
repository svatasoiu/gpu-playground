#ifndef UTILS_H__
#define UTILS_H__

// from udacity

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

typedef unsigned int ui;
typedef unsigned long long ull;

#define MAX_BLOCK_SIZE 1<<10
#define MAX_GRID_SIZE 1<<15
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
#define DISPLAY(x) std::cout << #x << ": " << (x) << std::endl

#ifdef NDEBUG
#define CHECK_ERR ;
#else
#define CHECK_ERR checkCudaErrors(cudaGetLastError());
#endif

template <class T>
struct Sorter {
  void (*sort)(const T *, T *, const size_t);
  std::string name;
};

#define INIT_SORT_FUNC(f, T) template void f<T>(const T *, T *, const size_t)
#define DECL_SORT_FUNC(f) template <class T> void f(const T * const, T * const, const size_t)
#define INIT_SORTER(f) { f, #f }

template <class T>
inline T round_up(T num, T divisor) {
    return (num + divisor - 1)/divisor;
}

// error checking stuff from udacity
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

template<typename T>
void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
  //check that the GPU result matches the CPU result
  for (size_t i = 0; i < numElem; ++i) {
    if (ref[i] != gpu[i]) {
      std::cerr << "Difference at pos " << i << std::endl;
      //the + is magic to convert char to int without messing
      //with other types
      std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
                 "\nGPU      : " << +gpu[i] << std::endl;
      exit(1);
    }
  }
}

template<typename T>
void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem, double eps1, double eps2) {
  assert(eps1 >= 0 && eps2 >= 0);
  unsigned long long totalDiff = 0;
  unsigned numSmallDifferences = 0;
  for (size_t i = 0; i < numElem; ++i) {
    //subtract smaller from larger in case of unsigned types
    T smaller = std::min(ref[i], gpu[i]);
    T larger = std::max(ref[i], gpu[i]);
    T diff = larger - smaller;
    if (diff > 0 && diff <= eps1) {
      numSmallDifferences++;
    }
    else if (diff > eps1) {
      std::cerr << "Difference at pos " << +i << " exceeds tolerance of " << eps1 << std::endl;
      std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
        "\nGPU      : " << +gpu[i] << std::endl;
      exit(1);
    }
    totalDiff += diff * diff;
  }
  double percentSmallDifferences = (double)numSmallDifferences / (double)numElem;
  if (percentSmallDifferences > eps2) {
    std::cerr << "Total percentage of non-zero pixel difference between the two images exceeds " << 100.0 * eps2 << "%" << std::endl;
    std::cerr << "Percentage of non-zero pixel differences: " << 100.0 * percentSmallDifferences << "%" << std::endl;
    exit(1);
  }
}

//Uses the autodesk method of image comparison
//Note the the tolerance here is in PIXELS not a percentage of input pixels
template<typename T>
void checkResultsAutodesk(const T* const ref, const T* const gpu, size_t numElem, double variance, size_t tolerance)
{

  size_t numBadPixels = 0;
  for (size_t i = 0; i < numElem; ++i) {
    T smaller = std::min(ref[i], gpu[i]);
    T larger = std::max(ref[i], gpu[i]);
    T diff = larger - smaller;
    if (diff > variance)
      ++numBadPixels;
  }

  if (numBadPixels > tolerance) {
    std::cerr << "Too many bad pixels in the image." << numBadPixels << "/" << tolerance << std::endl;
    exit(1);
  }
}

#endif
