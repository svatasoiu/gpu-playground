#pragma once

#include <string>
#include "options.h"

using namespace options;

namespace pricers {

template <typename T>
class BlackScholesEuroPricer : public Pricer<T> {
public:
  ~BlackScholesEuroPricer() {;}
  pricing_output<T> price(pricing_args<T>&);
  std::string getName();
};

}