#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "pricers.h"
#include "options.h"
#include "utils.h"
#include "timer.h"

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

  pricers::Pricer<test_type_t> *pricers[] = {
    new pricers::BlackScholesEuroPricer<test_type_t>(),
    new pricers::SimpleSequentialPricer<test_type_t>(),
    new pricers::SimpleParallelPricer<test_type_t>()
  };

  printf("Benchmarks run using data type of size %lu bytes\n", sizeof(test_type_t));

  std::string line;
  while (std::getline(inputFile, line)) {
    pricers::pricing_args<test_type_t> pargs;
    try {
      pargs = pricers::parse_args<test_type_t>(line);
    } catch (std::exception& err) {
      std::cerr << "Could not parse option: " << line << " [" << err.what() << "]" << std::endl;
      continue;
    }
    std::cout << "Option: " << pargs << std::endl;

    pricers::pricing_output<test_type_t> pricing_output;

    // run each pricer
    for (auto pricer : pricers) {
      GpuTimer timer;
      timer.Start();
      pricing_output = pricer->price(pargs);
      timer.Stop();
      cudaDeviceSynchronize(); 
      pricing_output.pricing_time = timer.Elapsed();
      std::cout << "  " << pricer->getName() << ": " << pricing_output << std::endl;
    }

    std::cout << std::endl;
  }

  for (auto pricer : pricers)
    delete pricer;

  return 0;
}
