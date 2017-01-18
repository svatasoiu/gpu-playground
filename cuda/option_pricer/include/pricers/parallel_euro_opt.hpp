#pragma once

#include "options.h"

template <typename T>
class SimpleParallelPricer : public options::Pricer {
public:
  pricing_output<T> price(pricing_args<T>&);
};