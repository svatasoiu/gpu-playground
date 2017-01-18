#pragma once

#include "options.h"

using namespace options;

template <typename T>
class SimpleSequentialPricer : public Pricer<T> {
public:
  pricing_output<T> price(pricing_args<T>&);
};