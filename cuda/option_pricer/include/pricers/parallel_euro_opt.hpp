#pragma once

#include "options.h"

using namespace options;

template <typename T>
class SimpleParallelPricer : public Pricer<T> {
public:
  ~SimpleParallelPricer() {;}
  pricing_output<T> price(pricing_args<T>&);
};