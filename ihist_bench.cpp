#include <ihist.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace ihist::bench {

namespace {

template <typename T, unsigned Bits = 8 * sizeof(T)>
auto generate_gaussian_data(std::size_t count, double stddev)
    -> std::vector<T> {
    static_assert(std::is_unsigned_v<T>);
    T const maximum = (1uLL << Bits) - 1;
    T const mean = maximum / 2;

    // std::normal_distribution requires stddev > 0.0
    if (stddev == 0.0) {
        return std::vector<T>(count, mean);
    }

    std::mt19937 engine;
    std::normal_distribution<double> dist(static_cast<double>(mean), stddev);

    // Generating normally distributed random numbers is slow. Since this is
    // just a benchmark, we cheat a bit by repeating a pattern.

    std::vector<T> population(1 << 16);
    std::generate(population.begin(), population.end(), [&] {
        for (;;) {
            double const v = std::round(dist(engine));
            if (v >= 0.0 && v <= maximum) {
                return static_cast<T>(v);
            }
        }
    });

    std::vector<T> data;
    data.reserve(count / population.size() * population.size());
    while (data.size() < count) {
        data.insert(data.end(), population.begin(), population.end());
    }
    data.resize(count);
    return data;
}

template <typename T, auto Hist, unsigned Bits, std::size_t Stride = 1,
          std::size_t Component0Offset = 0, std::size_t... ComponentOffsets>
void hist_gauss(benchmark::State &state) {
    constexpr auto NCOMPONENTS = 1 + sizeof...(ComponentOffsets);
    auto const stddev = static_cast<double>(state.range(0));
    auto const size = state.range(1);
    auto const data = generate_gaussian_data<T, Bits>(size * Stride, stddev);
    for ([[maybe_unused]] auto _ : state) {
        std::array<std::uint32_t, NCOMPONENTS * (1 << Bits)> hist{};
        Hist(data.data(), size, hist.data());
        benchmark::DoNotOptimize(hist);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size *
                            Stride * sizeof(T));
    state.counters["samples_per_second"] = benchmark::Counter(
        static_cast<int64_t>(state.iterations()) * size * NCOMPONENTS,
        benchmark::Counter::kIsRate);
    state.counters["pixels_per_second"] =
        benchmark::Counter(static_cast<int64_t>(state.iterations()) * size,
                           benchmark::Counter::kIsRate);
}

// A standard deviation of 0 produces constant data, and a large one closely
// approximates uniformly random data. Generally data that is narrowly
// distributed is more challenging due to store-to-load forwarding latencies on
// the same bin or cache line.
template <unsigned Bits>
std::vector<std::int64_t> const stddevs{0, 1, 1 << (Bits + 1)};

// Quite a large data size (16Mi = 1 << 26) is needed for the throughput to
// plateau, especially for multithreaded. Currently, we use a fixed pixel count
// regardless of format.
template <typename T, std::size_t Stride = 1>
std::vector<std::int64_t> const data_sizes{20'000'000}; // 20 megapixels

using u8 = std::uint8_t;
using u16 = std::uint16_t;

} // namespace

#define HIST_BENCH(bits, filt, T, P, threading)                               \
    constexpr tuning_parameters                                               \
        tune_##bits##b_##T##_##filt##_striped##P##_##threading{               \
            default_tuning_parameters<T, bits>.prefer_branchless,             \
            1 << P,                                                           \
            default_tuning_parameters<T, bits>.n_unroll,                      \
            default_tuning_parameters<T, bits>.mt_grain_size,                 \
        };                                                                    \
    BENCHMARK(hist_gauss<                                                     \
                  T,                                                          \
                  hist_##filt##_striped_##threading<                          \
                      tune_##bits##b_##T##_##filt##_striped##P##_##threading, \
                      T, bits>,                                               \
                  bits>)                                                      \
        ->Name(#bits "b-" #T "-" #filt "-striped" #P "-" #threading)          \
        ->ArgsProduct({stddevs<bits>, data_sizes<T>});

#define HIST_BENCH_RGB(bits, filt, T, P, threading)                           \
    constexpr tuning_parameters                                               \
        tune_rgb_##bits##b_##T##_##filt##_striped##P##_##threading{           \
            default_tuning_parameters<T, bits>.prefer_branchless,             \
            1 << P,                                                           \
            default_tuning_parameters<T, bits>.n_unroll,                      \
            default_tuning_parameters<T, bits>.mt_grain_size,                 \
        };                                                                    \
    BENCHMARK(                                                                \
        hist_gauss<                                                           \
            T,                                                                \
            hist_##filt##_striped_##threading<                                \
                tune_rgb_##bits##b_##T##_##filt##_striped##P##_##threading,   \
                T, bits, 0, 3, 0, 1, 2>,                                      \
            bits, 3, 0, 1, 2>)                                                \
        ->Name("rgb-" #bits "b-" #T "-" #filt "-striped" #P "-" #threading)   \
        ->ArgsProduct({stddevs<bits>, data_sizes<T, 3>});

#define HIST_BENCH_RGB_(bits, filt, T, P, threading)                          \
    constexpr tuning_parameters                                               \
        tune_rgbx_##bits##b_##T##_##filt##_striped##P##_##threading{          \
            default_tuning_parameters<T, bits>.prefer_branchless,             \
            1 << P,                                                           \
            default_tuning_parameters<T, bits>.n_unroll,                      \
            default_tuning_parameters<T, bits>.mt_grain_size,                 \
        };                                                                    \
    BENCHMARK(                                                                \
        hist_gauss<                                                           \
            T,                                                                \
            hist_##filt##_striped_##threading<                                \
                tune_rgbx_##bits##b_##T##_##filt##_striped##P##_##threading,  \
                T, bits, 0, 4, 0, 1, 2>,                                      \
            bits, 4, 0, 1, 2>)                                                \
        ->Name("rgb_-" #bits "b-" #T "-" #filt "-striped" #P "-" #threading)  \
        ->ArgsProduct({stddevs<bits>, data_sizes<T, 3>});

#define HIST_BENCH_SET(bits, filt, T)                                         \
    HIST_BENCH(bits, filt, T, 0, st)                                          \
    HIST_BENCH(bits, filt, T, 1, st)                                          \
    HIST_BENCH(bits, filt, T, 2, st)                                          \
    HIST_BENCH(bits, filt, T, 3, st)                                          \
    HIST_BENCH(bits, filt, T, 0, mt)                                          \
    HIST_BENCH(bits, filt, T, 1, mt)                                          \
    HIST_BENCH(bits, filt, T, 2, mt)                                          \
    HIST_BENCH(bits, filt, T, 3, mt)

#define HIST_BENCH_SET_RGB(bits, filt, T)                                     \
    HIST_BENCH_RGB(bits, filt, T, 0, st)                                      \
    HIST_BENCH_RGB(bits, filt, T, 1, st)                                      \
    HIST_BENCH_RGB(bits, filt, T, 0, mt)                                      \
    HIST_BENCH_RGB(bits, filt, T, 1, mt)                                      \
    HIST_BENCH_RGB_(bits, filt, T, 0, st)                                     \
    HIST_BENCH_RGB_(bits, filt, T, 1, st)                                     \
    HIST_BENCH_RGB_(bits, filt, T, 0, mt)                                     \
    HIST_BENCH_RGB_(bits, filt, T, 1, mt)

HIST_BENCH_SET(8, unfiltered, u8)
HIST_BENCH_SET(8, unfiltered, u16)
HIST_BENCH_SET(9, unfiltered, u16)
HIST_BENCH_SET(9, filtered, u16)
HIST_BENCH_SET(12, unfiltered, u16)
HIST_BENCH_SET(12, filtered, u16)
HIST_BENCH_SET(14, unfiltered, u16)
HIST_BENCH_SET(14, filtered, u16)
HIST_BENCH_SET(16, unfiltered, u16)

HIST_BENCH_SET_RGB(8, unfiltered, u8)
HIST_BENCH_SET_RGB(12, unfiltered, u16)
HIST_BENCH_SET_RGB(12, filtered, u16)
HIST_BENCH_SET_RGB(14, unfiltered, u16)
HIST_BENCH_SET_RGB(14, filtered, u16)
HIST_BENCH_SET_RGB(16, unfiltered, u16)

} // namespace ihist::bench

BENCHMARK_MAIN(); // NOLINT
