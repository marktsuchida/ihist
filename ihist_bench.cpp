#include <ihist.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace ihist::bench {

namespace {

template <typename T> constexpr auto midpoint_value() -> T {
    if constexpr (std::is_signed_v<T>) {
        return T(0);
    } else {
        return std::numeric_limits<T>::max() / 2;
    }
}

template <typename T>
auto generate_gaussian_data(std::size_t count, double stddev)
    -> std::vector<T> {
    T const minimum = std::numeric_limits<T>::min();
    T const maximum = std::numeric_limits<T>::max();
    T const mean = midpoint_value<T>();

    std::mt19937 engine;
    std::normal_distribution<double> dist(static_cast<double>(mean), stddev);

    std::vector<T> data;
    data.resize(count);
    std::generate(data.begin(), data.end(), [&] {
        for (;;) {
            double v = dist(engine);
            if (v >= minimum && v <= maximum) {
                return static_cast<T>(v);
            }
        }
    });
    return data;
}

template <auto Hist8> void hist8_gauss(benchmark::State &state) {
    auto const stddev = static_cast<double>(state.range(0));
    auto const size = state.range(1);
    auto const data = generate_gaussian_data<std::uint8_t>(size, stddev);
    for ([[maybe_unused]] auto _ : state) {
        std::array<std::uint32_t, 256> hist{};
        Hist8(data.data(), size, hist.data());
        benchmark::DoNotOptimize(hist);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size *
                            sizeof(std::uint8_t));
}

// A standard deviation of 0 produces constant data, and 500 closely
// approximates uniformly random data. Generally data that is narrowly
// distributed is more challenging due to store-to-load forwarding latencies on
// the same bin or cache line.
std::vector<std::int64_t> const stddevs{0, 2, 500};

// Quite a large data size (16Mi = 1 << 26) is needed for the throughput to
// plateau. But the trend over algorithms and input data stay the same.
std::vector<std::int64_t> const data_sizes{1 << 20};

} // namespace

BENCHMARK(hist8_gauss<hist8_naive>)->ArgsProduct({stddevs, data_sizes});

BENCHMARK(hist8_gauss<hist8_striped<3>>)->ArgsProduct({stddevs, data_sizes});

BENCHMARK(hist8_gauss<hist8_naive_mt>)->ArgsProduct({stddevs, data_sizes});

BENCHMARK(hist8_gauss<hist8_striped_mt<3>>)
    ->ArgsProduct({stddevs, data_sizes});

} // namespace ihist::bench

BENCHMARK_MAIN(); // NOLINT