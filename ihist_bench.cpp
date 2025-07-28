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

BENCHMARK(hist_gauss<u8, hist_unfiltered_naive<u8>>)
    ->ArgsProduct({stddevs<8>, data_sizes<u8>});

BENCHMARK(hist_gauss<u8, hist_unfiltered_striped<3, u8>>)
    ->ArgsProduct({stddevs<8>, data_sizes<u8>});

BENCHMARK(hist_gauss<u8, hist_unfiltered_naive_mt<u8>>)
    ->ArgsProduct({stddevs<8>, data_sizes<u8>});

BENCHMARK(hist_gauss<u8, hist_unfiltered_striped_mt<3, u8>>)
    ->ArgsProduct({stddevs<8>, data_sizes<u8>});

BENCHMARK(hist_gauss<u16, hist_unfiltered_naive<u16, 12>, 12>)
    ->ArgsProduct({stddevs<12>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_unfiltered_striped<2, u16, 12>, 12>)
    ->ArgsProduct({stddevs<12>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_unfiltered_naive_mt<u16, 12>, 12>)
    ->ArgsProduct({stddevs<12>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_unfiltered_striped_mt<2, u16, 12>, 12>)
    ->ArgsProduct({stddevs<12>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_filtered_naive<u16, 12>, 12>)
    ->ArgsProduct({stddevs<12>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_filtered_striped<2, u16, 12>, 12>)
    ->ArgsProduct({stddevs<12>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_filtered_naive_mt<u16, 12>, 12>)
    ->ArgsProduct({stddevs<12>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_filtered_striped_mt<2, u16, 12>, 12>)
    ->ArgsProduct({stddevs<12>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_unfiltered_naive<u16>>)
    ->ArgsProduct({stddevs<16>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_unfiltered_striped<2, u16>>)
    ->ArgsProduct({stddevs<16>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_unfiltered_radixpartition<8, u16>>)
    ->ArgsProduct({stddevs<16>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_unfiltered_naive_mt<u16>>)
    ->ArgsProduct({stddevs<16>, data_sizes<u16>});

BENCHMARK(hist_gauss<u16, hist_unfiltered_striped_mt<2, u16>>)
    ->ArgsProduct({stddevs<16>, data_sizes<u16>});

} // namespace ihist::bench

BENCHMARK_MAIN(); // NOLINT