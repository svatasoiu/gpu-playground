#pragma once

#include "utils/stats.h"

#include <random>
#include <type_traits>
#include <vector>

// generic monte carlo simulator 
// produces samples, then combines them into estimate
namespace monte_carlo {

template <class T>
class SampleGenerator {
public:
    SampleGenerator() {}
    virtual ~SampleGenerator() {}

    virtual void generateSample(T&) = 0;
};

template <class T>
class RandomWalkGenerator : public SampleGenerator<T> {
    // static assert T has some sort of ordering (like vector)
private:
    using ValueT = typename T::value_type;
    static_assert(std::is_floating_point<ValueT>::value,
        "RandomWalkGenerator requires a container of floating point type");

    const ValueT S0, t, r, vol;
    const size_t path_len;

    std::default_random_engine generator;
    std::normal_distribution<ValueT> distribution;
public:
    RandomWalkGenerator(ValueT S0, ValueT t, ValueT r, ValueT vol, size_t path_len);
    ~RandomWalkGenerator();

    void generateSample(T&);
};

template <class SampleT, class OutputT>
class PayoffCalculator {
public:
    PayoffCalculator() {}
    virtual ~PayoffCalculator() {}
    
    virtual OutputT calculate(const SampleT&) = 0;
};

template <class SampleT>
class EuropeanPathPayoff : public PayoffCalculator<SampleT, typename SampleT::value_type> {
private:
    using ValueT = typename SampleT::value_type;

    const bool is_call;
    const ValueT discount;
    const ValueT K;
public:
    EuropeanPathPayoff(const bool is_call, const ValueT discount, const ValueT K)
     : is_call(is_call), discount(discount), K(K) {}
    ~EuropeanPathPayoff() {}
    
    ValueT calculate(const SampleT& path) {
        return discount * (is_call ? std::max(path.back() - K, (ValueT)0) : std::max(K - path.back(), (ValueT)0));
    }
};

template <typename T>
struct MonteCarloResult {
	T estimate;
	T stderr;
};

template <class SampleT, class OutputT>
class MonteCarlo {
    // static assert that outputT is numeric
protected:
    SampleGenerator<SampleT>& generator;
    PayoffCalculator<SampleT,OutputT>& payoff;
public:
    MonteCarlo(SampleGenerator<SampleT> &g, PayoffCalculator<SampleT,OutputT> &p) 
        : generator(g), payoff(p) { ; }

    virtual ~MonteCarlo() { ; }

    virtual MonteCarloResult<OutputT> estimate(size_t) = 0; // # trials
};

template <class SampleT, class OutputT>
class SimpleMonteCarlo : public MonteCarlo<SampleT, OutputT> {
public:
    SimpleMonteCarlo(SampleGenerator<SampleT> &g, PayoffCalculator<SampleT,OutputT> &p) 
        : MonteCarlo<SampleT, OutputT>(g, p) { ; }
    ~SimpleMonteCarlo() {}

    MonteCarloResult<OutputT> estimate(size_t num_trials) {
        SampleT sample;
        std::vector<OutputT> estimates(num_trials);
        
        for (size_t i = 0; i < num_trials; ++i) {
            this->generator.generateSample(sample);
            estimates[i] = this->payoff.calculate(sample);
        }

        return {
            stats::mean(estimates), 
            stats::standard_error(estimates)
        };
    }
};

/*
template <class T>
class AntitheticMonteCarlo : public MonteCarlo<T> {
public:
    MonteCarlo(SampleGenerator<T>, Payoff<T>);
    virtual ~MonteCarlo();

    virtual estimate(size_t);
}
*/
}

// template implementation
#include "utils/random_walk.tcc"