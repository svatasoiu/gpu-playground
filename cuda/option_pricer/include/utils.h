#ifndef UTILS_H__
#define UTILS_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdio.h>

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

#define INIT_PRICER(p, T) template class f<T>

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

// memory info
// taken from (https://devtalk.nvidia.com/default/topic/389173/cumemgetinfo-/)
static void printStats(CUdevprop &prop)
{
  printf(" warp size: %d\n", prop.SIMDWidth);
  printf(" sharedMemPerBlock : %d KB\n", prop.sharedMemPerBlock/1024);
  printf(" clockRate: %d MHz\n", prop.clockRate/1024);
  printf(" Maximum Theo Bandwidth: %d GB/s\n", prop.clockRate*128/1024/1024);
}

static void displayDeviceInfo() {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1000);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
  printf("\n");
}

// printing stuff
#include <cstdarg>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

static void print_test_case_result(bool success, const char *format, ...) {
  printf(success ? ANSI_COLOR_GREEN : ANSI_COLOR_RED);
  va_list vl;
  va_start(vl, format);
  vprintf(format, vl);
  va_end(vl);
  printf(ANSI_COLOR_RESET);
}

#endif
