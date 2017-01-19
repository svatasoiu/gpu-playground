#pragma once

#include "options.h"

using namespace options;

namespace pricers {

template <typename T>
class SimpleSequentialPricer : public Pricer<T> {
public:
  ~SimpleSequentialPricer() {;}
  pricing_output<T> price(pricing_args<T>&);
  std::string getName();
};

}