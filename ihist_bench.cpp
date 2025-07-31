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

template <typename T, unsigned BITS = 8 * sizeof(T)>
auto generate_gaussian_data(std::size_t count, double stddev)
    -> std::vector<T> {
    static_assert(std::is_unsigned_v<T>);
    T const maximum = (1uLL << BITS) - 1;
    T const mean = maximum / 2;

    // std::normal_distribution requires stddev > 0.0
    if (stddev == 0.0) {
        return std::vector<T>(count, mean);
    }

    std::mt19937 engine;
    std::normal_distribution<double> dist(static_cast<double>(mean), stddev);

    std::vector<T> data;
    data.resize(count);
    std::generate(data.begin(), data.end(), [&] {
        for (;;) {
            double v = std::round(dist(engine));
            if (v >= 0.0 && v <= maximum) {
                return static_cast<T>(v);
            }
        }
    });
    return data;
}

template <typename T, auto Hist, unsigned BITS = 8 * sizeof(T)>
void hist_gauss(benchmark::State &state) {
    auto const stddev = static_cast<double>(state.range(0));
    auto const size = state.range(1);
    auto const data = generate_gaussian_data<T, BITS>(size, stddev);
    for ([[maybe_unused]] auto _ : state) {
        std::array<std::uint32_t, (1 << (8 * sizeof(T)))> hist{};
        Hist(data.data(), size, hist.data());
        benchmark::DoNotOptimize(hist);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size *
                            sizeof(T));
    state.counters["pixels_per_second"] =
        benchmark::Counter(static_cast<int64_t>(state.iterations()) * size,
                           benchmark::Counter::kIsRate);
}

// A standard deviation of 0 produces constant data, and a large one closely
// approximates uniformly random data. Generally data that is narrowly
// distributed is more challenging due to store-to-load forwarding latencies on
// the same bin or cache line.
template <unsigned BITS>
std::vector<std::int64_t> const stddevs{0, 1 << (BITS + 1)};

// Quite a large data size (16Mi = 1 << 26) is needed for the throughput to
// plateau. But the trend over algorithms and input data stay the same.
template <typename T>
std::vector<std::int64_t> const data_sizes{1 << (20 - sizeof(T))};

using u8 = std::uint8_t;
using u16 = std::uint16_t;

} // namespace

#define HIST_BENCH(bits, filt, T, P, threading)                               \
    BENCHMARK(                                                                \
        hist_gauss<T, hist_##filt##_striped_##threading<P, T, bits>, bits>)   \
        ->Name(#bits "b-" #T "-" #filt "-striped" #P "-" #threading)          \
        ->ArgsProduct({stddevs<bits>, data_sizes<T>});

#define HIST_BENCH_SET(bits, filt, T)                                         \
    HIST_BENCH(bits, filt, T, 0, st)                                          \
    HIST_BENCH(bits, filt, T, 1, st)                                          \
    HIST_BENCH(bits, filt, T, 2, st)                                          \
    HIST_BENCH(bits, filt, T, 3, st)                                          \
    HIST_BENCH(bits, filt, T, 0, mt)                                          \
    HIST_BENCH(bits, filt, T, 1, mt)                                          \
    HIST_BENCH(bits, filt, T, 2, mt)                                          \
    HIST_BENCH(bits, filt, T, 3, mt)

HIST_BENCH_SET(8, unfiltered, u8)
HIST_BENCH_SET(8, unfiltered, u16)
HIST_BENCH_SET(9, unfiltered, u16)
HIST_BENCH_SET(9, filtered, u16)
HIST_BENCH_SET(12, unfiltered, u16)
HIST_BENCH_SET(12, filtered, u16)
HIST_BENCH_SET(14, unfiltered, u16)
HIST_BENCH_SET(14, filtered, u16)
HIST_BENCH_SET(16, unfiltered, u16)

} // namespace ihist::bench

BENCHMARK_MAIN(); // NOLINT
