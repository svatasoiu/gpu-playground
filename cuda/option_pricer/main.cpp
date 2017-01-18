#include <cassert>
#include <cstring>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "pricers.h"
#include "options.h"
#include "utils.h"

typedef double test_type_t;

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Error: need to supply a file from which to read option parameters\nUSAGE: %s <input file name>\n", argv[0]);
		exit(1);
	}

#ifndef NDEBUG
  int err;
  char host[1024];
  if ((err = gethostname(host, 1024)) != 0) {
    printf("Error getting hostname\n");
  } else {
    printf("Running on host: %s\n", host);
  }
  displayDeviceInfo();
#endif

  std::ifstream inputFile(argv[1]);

  if(!inputFile) {
    // failed to open file
    printf("Error: failed to open %s\n", argv[1]);
    exit(1);
  }
  
  SimpleSequentialPricer<test_type_t> ssp();
  SimpleParallelPricer<test_type_t>   spp();

  options::Pricer<test_type_t> *pricers[] = {
    &ssp, &spp
  };

  printf("Benchmarks run using data type of size %lu bytes\n", sizeof(test_type_t));

  std::string line;
  while (std::getline(inputFile, line)) {
    auto pargs = options::parse_args<test_type_t>(line);
    pargs.display();

    options::pricing_output<test_type_t> pricing_output;

    // run each pricer
    for (auto pricer : pricers) {
      pricing_output = pricer->price(pargs);
      cudaDeviceSynchronize(); 
      pricing_output.display();
      // bool success = check_matrices(h_mat, h_out, N);
      // print_test_case_result(success, "%6.2f ms ", transposing_time);
      // print_test_case_result(success, "(%6.2f GB/s) ", 2 * matrixSize / transposing_time / 1000000.f );
    }

    printf("\n");
  }

  return 0;
}
