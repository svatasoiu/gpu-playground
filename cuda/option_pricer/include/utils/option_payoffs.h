#pragma once

#include <stdlib.h>

template <class T>
static inline T euro_call_option_payoff(T S, T K) {
    return std::max(S - K, 0);
}

template <class T>
static inline T euro_put_option_payoff(T S, T K) {
    return std::max(K - S, 0);
}