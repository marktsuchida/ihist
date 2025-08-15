// This header requires the following macros to be predefined:
// BM_NAME_PREFIX (mono, rgb, rgbx)
// BM_STRIDE_COMPONENTS
// BM_BITS
// BM_FILTERING (unfiltered, filtered)
// BM_THREADING (st, mt)

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

template <typename T, unsigned Bits = 8 * sizeof(T)>
auto generate_data(std::size_t count, float spread_frac) -> std::vector<T> {
    static_assert(std::is_unsigned_v<T>);
    static_assert(Bits < 8 * sizeof(std::size_t));
    std::size_t const maximum = (1uLL << Bits) - 1;
    std::size_t const mean = maximum / 2;

    if (spread_frac <= 0.0f) {
        return std::vector<T>(count, mean);
    }

    auto const half_spread = std::clamp<std::size_t>(
        std::llroundf(0.5f * spread_frac * static_cast<float>(maximum)), 0,
        mean);

    std::mt19937 engine;
    std::uniform_int_distribution<std::size_t> dist(mean - half_spread,
                                                    mean + half_spread);

    // Since this is just for a benchmark, we cheat, for speed, by repeating a
    // pattern.
    std::vector<T> population(1 << 16);

    std::generate(population.begin(), population.end(), [&] {
        for (;;) {
            auto const v = dist(engine);
            if (v <= maximum) {
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

using u8 = std::uint8_t;
using u16 = std::uint16_t;

template <std::size_t Bits>
using bits_type = std::conditional_t<(Bits > 8), u16, u8>;

template <auto Hist, unsigned Bits, std::size_t Stride = 1,
          std::size_t Component0Offset = 0, std::size_t... ComponentOffsets>
void bm_hist(benchmark::State &state) {
    using T = bits_type<Bits>;
    constexpr auto NCOMPONENTS = 1 + sizeof...(ComponentOffsets);
    auto const size = state.range(0);
    auto const spread_frac = static_cast<float>(state.range(1)) / 100.0f;
    auto const data = generate_data<T, Bits>(size * Stride, spread_frac);
    for ([[maybe_unused]] auto _ : state) {
        std::array<std::uint32_t, NCOMPONENTS * (1 << Bits)> hist{};
        auto const *d = data.data();
        auto *h = hist.data();
#ifdef __clang__
        // If the function gets inlined here, it may be able to receive
        // additional optimizations that otherwise would not be possible (e.g.,
        // based on alignment of data and hist). We do not want that for a fair
        // benchmark.
        [[clang::noinline]]
#endif
        Hist(d, size, h);
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

// The spread of the data affects performance: if narrow, a simple
// implementation will be bound by store-to-load forwarding latencyes due to
// incrementing a bin on the same cache line in close succession. Striped
// implementations may slow down for a wide distribution due to the larger
// working set size.
// 6% and 25% are useful for comparing 16-bit performance with 12/14-bit
// performance.
template <unsigned Bits>
std::vector<std::int64_t> const spread_pcts{0, 1, 6, 25, 100};

// Quite a large data size (16Mi = 1 << 26) is needed for the throughput to
// plateau, especially for multithreaded. Currently, we use a fixed pixel count
// regardless of format.
std::vector<std::int64_t> const data_sizes{20'000'000}; // 20 megapixels

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define TUNING_NAME(branchless, stripes, unrolls, grainsize)                  \
    tuning_bl##branchless##_st##stripes##_un##unrolls##_gr##grainsize

#define DEFINE_TUNING(branchless, stripes, unrolls, grainsize)                \
    constexpr tuning_parameters TUNING_NAME(branchless, stripes, unrolls,     \
                                            grainsize){branchless, stripes,   \
                                                       unrolls, grainsize};

#define DEFINE_HISTBM(branchless, stripes, unrolls, grainsize, bits, filt,    \
                      thd)                                                    \
    DEFINE_TUNING(branchless, stripes, unrolls, grainsize)                    \
    BENCHMARK(                                                                \
        bm_hist<hist_##filt##_striped_##thd<                                  \
                    TUNING_NAME(branchless, stripes, unrolls, grainsize),     \
                    bits_type<bits>, bits, 0, BM_STRIDE_COMPONENTS>,          \
                bits, BM_STRIDE_COMPONENTS>)                                  \
        ->Name(TOSTRING(                                                      \
            BM_NAME_PREFIX) "/bits:" #bits "/filt:" #filt "/threading:" #thd  \
                            "/branchless:" #branchless "/stripes:" #stripes   \
                            "/unrolls:" #unrolls "/grainsize:" #grainsize)    \
        ->MeasureProcessCPUTime()                                             \
        ->UseRealTime()                                                       \
        ->ArgNames({"size", "spread"})                                        \
        ->ArgsProduct({data_sizes, spread_pcts<bits>});

#define DEFINE_HISTBM_BRANCHLESS(stripes, unrolls, grainsize, bits, filt,     \
                                 thd)                                         \
    DEFINE_HISTBM(false, stripes, unrolls, grainsize, bits, filt, thd)        \
    DEFINE_HISTBM(true, stripes, unrolls, grainsize, bits, filt, thd)

#define DEFINE_HISTBM_STRIPES(unrolls, grainsize, bits, filt, thd)            \
    DEFINE_HISTBM_BRANCHLESS(1, unrolls, grainsize, bits, filt, thd)          \
    DEFINE_HISTBM_BRANCHLESS(2, unrolls, grainsize, bits, filt, thd)          \
    DEFINE_HISTBM_BRANCHLESS(4, unrolls, grainsize, bits, filt, thd)          \
    DEFINE_HISTBM_BRANCHLESS(8, unrolls, grainsize, bits, filt, thd)

#define DEFINE_HISTBM_UNROLLS(grainsize, bits, filt, thd)                     \
    DEFINE_HISTBM_STRIPES(1, grainsize, bits, filt, thd)                      \
    DEFINE_HISTBM_STRIPES(2, grainsize, bits, filt, thd)                      \
    DEFINE_HISTBM_STRIPES(4, grainsize, bits, filt, thd)                      \
    DEFINE_HISTBM_STRIPES(8, grainsize, bits, filt, thd)

#define DEFINE_HIST_BENCHMARKS(bits, filt, thd)                               \
    DEFINE_HISTBM_UNROLLS(16384, bits, filt, thd)                             \
    DEFINE_HISTBM_UNROLLS(65536, bits, filt, thd)                             \
    DEFINE_HISTBM_UNROLLS(262144, bits, filt, thd)

DEFINE_HIST_BENCHMARKS(BM_BITS, BM_FILTERING, BM_THREADING)

} // namespace ihist::bench