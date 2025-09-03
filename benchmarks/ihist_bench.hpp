/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

// This header generates all the benchmarks for a given data format. A separate
// translation unit is used for each data format, because generating all the
// tests at once overwhelms the compiler.
//
// The following macros must be defined before including this header:
// BM_NAME_PREFIX (mono, rgb, rgbx)
// BM_STRIDE_COMPONENTS
// BM_BITS
// BM_MULTITHREADING (0, 1)

#ifdef IHIST_BENCH_HPP_INCLUDED
#error Cannot include this file twice in the same translation unit
#endif
#define IHIST_BENCH_HPP_INCLUDED 1

#include "ihist.hpp"

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

using u8 = std::uint8_t;
using u16 = std::uint16_t;

template <std::size_t Bits>
using bits_type = std::conditional_t<(Bits > 8), u16, u8>;

enum class roi_type {
    one_d,
    two_d,
};

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

inline auto generate_circle_mask(std::intptr_t width, std::intptr_t height)
    -> std::vector<u8> {
    std::vector<u8> mask(width * height);
    auto const center_x = width / 2;
    auto const center_y = height / 2;
    for (std::intptr_t y = 0; y < height; ++y) {
        for (std::intptr_t x = 0; x < width; ++x) {
            auto const xx =
                (x - center_x) * (x - center_x) * center_y * center_y;
            auto const yy =
                (y - center_y) * (y - center_y) * center_x * center_x;
            bool is_inside =
                xx + yy < center_x * center_x * center_y * center_y;
            mask[x + y * width] = static_cast<u8>(is_inside);
        }
    }
    return mask;
}

template <auto Hist, unsigned Bits, std::size_t Stride = 1,
          std::size_t Component0Offset = 0, std::size_t... ComponentOffsets>
void bm_hist(benchmark::State &state) {
    using T = bits_type<Bits>;
    constexpr auto NCOMPONENTS = 1 + sizeof...(ComponentOffsets);
    auto const width = state.range(0);
    auto const height = state.range(0);
    auto const size = width * height;
    auto const spread_frac = static_cast<float>(state.range(1)) / 100.0f;
    auto const grain_size = static_cast<std::size_t>(state.range(2));
    auto const data = generate_data<T, Bits>(size * Stride, spread_frac);
    auto const mask = generate_circle_mask(width, height);
    for ([[maybe_unused]] auto _ : state) {
        std::array<std::uint32_t, NCOMPONENTS * (1 << Bits)> hist{};
        auto const *d = data.data();
        auto const *m = mask.data();
        auto *h = hist.data();
        Hist(d, m, size, h, grain_size);
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

template <auto Hist, roi_type RoiType, unsigned Bits, std::size_t Stride = 1,
          std::size_t Component0Offset = 0, std::size_t... ComponentOffsets>
void bm_histxy(benchmark::State &state) {
    using T = bits_type<Bits>;
    constexpr auto NCOMPONENTS = 1 + sizeof...(ComponentOffsets);
    auto const width = state.range(0);
    auto const height = state.range(0);
    auto const size = width * height;
    auto const spread_frac = static_cast<float>(state.range(1)) / 100.0f;
    auto const grain_size = static_cast<std::size_t>(state.range(2));
    // For now, ROI is full image.
    auto const roi_size = width * height;
    auto const data = generate_data<T, Bits>(size * Stride, spread_frac);
    auto const mask = generate_circle_mask(width, height);
    for ([[maybe_unused]] auto _ : state) {
        std::array<std::uint32_t, NCOMPONENTS * (1 << Bits)> hist{};
        auto const *d = data.data();
        auto const *m = mask.data();
        auto *h = hist.data();
        Hist(d, m, width, height, 0, 0, width, height, h, grain_size);
        benchmark::DoNotOptimize(hist);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            roi_size * Stride * sizeof(T));
    state.counters["samples_per_second"] = benchmark::Counter(
        static_cast<int64_t>(state.iterations()) * roi_size * NCOMPONENTS,
        benchmark::Counter::kIsRate);
    state.counters["pixels_per_second"] =
        benchmark::Counter(static_cast<int64_t>(state.iterations()) * roi_size,
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

// For single-threaded, performance starts to drop when the data no longer fits
// in the last-level cache, but that is not an effect we are particularly
// interested in (because there is not much we can do to prevent it). For
// multi-threaded, it is hard to reach a performnce plateau, even with hundreds
// of megapixels. 16 Mi (1 << 24) is probably a reasonable compromize if
// looking at a single size (and the image size of large-ish CMOS chips).
// data_sizes is the width and height of the data.
std::vector<std::int64_t> const data_sizes{1 << 12};

std::vector<std::int64_t> const mt_grain_sizes{16384, 65536, 262144};
std::vector<std::int64_t> const st_grain_sizes{0};

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define TUNING_NAME(stripes, unrolls) tuning_st##stripes##_un##unrolls

#define DEFINE_TUNING(stripes, unrolls)                                       \
    inline constexpr tuning_parameters TUNING_NAME(stripes, unrolls){         \
        stripes, unrolls};

#define BENCH_NAME(stripes, unrolls, bits, roitype, mask)                     \
    TOSTRING(BM_NAME_PREFIX)                                                  \
    "/bits:" #bits "/" #roitype "/mask:" #mask                                \
    "/mt:" TOSTRING(BM_MULTITHREADED) "/stripes:" #stripes                    \
                                      "/unrolls:" #unrolls

#define DEFINE_HISTBM(stripes, unrolls, grainsizes, bits, thd)                \
    DEFINE_TUNING(stripes, unrolls)                                           \
    BENCHMARK(bm_hist<hist_striped_##thd<TUNING_NAME(stripes, unrolls),       \
                                         bits_type<bits>, false, bits, 0,     \
                                         BM_STRIDE_COMPONENTS>,               \
                      bits, BM_STRIDE_COMPONENTS>)                            \
        ->Name(BENCH_NAME(stripes, unrolls, bits, roi_type::one_d, 0))        \
        ->MeasureProcessCPUTime()                                             \
        ->UseRealTime()                                                       \
        ->ArgNames({"size", "spread", "grainsize"})                           \
        ->ArgsProduct({data_sizes, spread_pcts<bits>, grainsizes});           \
    BENCHMARK(bm_hist<hist_striped_##thd<TUNING_NAME(stripes, unrolls),       \
                                         bits_type<bits>, true, bits, 0,      \
                                         BM_STRIDE_COMPONENTS>,               \
                      bits, BM_STRIDE_COMPONENTS>)                            \
        ->Name(BENCH_NAME(stripes, unrolls, bits, roi_type::one_d, 1))        \
        ->MeasureProcessCPUTime()                                             \
        ->UseRealTime()                                                       \
        ->ArgNames({"size", "spread", "grainsize"})                           \
        ->ArgsProduct({data_sizes, spread_pcts<bits>, grainsizes});           \
    BENCHMARK(bm_histxy<histxy_striped_##thd<TUNING_NAME(stripes, unrolls),   \
                                             bits_type<bits>, false, bits, 0, \
                                             BM_STRIDE_COMPONENTS>,           \
                        roi_type::two_d, bits, BM_STRIDE_COMPONENTS>)         \
        ->Name(BENCH_NAME(stripes, unrolls, bits, roi_type::two_d, 0))        \
        ->MeasureProcessCPUTime()                                             \
        ->UseRealTime()                                                       \
        ->ArgNames({"size", "spread", "grainsize"})                           \
        ->ArgsProduct({data_sizes, spread_pcts<bits>, grainsizes});           \
    BENCHMARK(bm_histxy<histxy_striped_##thd<TUNING_NAME(stripes, unrolls),   \
                                             bits_type<bits>, true, bits, 0,  \
                                             BM_STRIDE_COMPONENTS>,           \
                        roi_type::two_d, bits, BM_STRIDE_COMPONENTS>)         \
        ->Name(BENCH_NAME(stripes, unrolls, bits, roi_type::two_d, 1))        \
        ->MeasureProcessCPUTime()                                             \
        ->UseRealTime()                                                       \
        ->ArgNames({"size", "spread", "grainsize"})                           \
        ->ArgsProduct({data_sizes, spread_pcts<bits>, grainsizes});

#define DEFINE_HISTBM_STRIPES(unrolls, grainsizes, bits, thd)                 \
    DEFINE_HISTBM(1, unrolls, grainsizes, bits, thd)                          \
    DEFINE_HISTBM(2, unrolls, grainsizes, bits, thd)                          \
    DEFINE_HISTBM(4, unrolls, grainsizes, bits, thd)                          \
    DEFINE_HISTBM(8, unrolls, grainsizes, bits, thd)                          \
    DEFINE_HISTBM(16, unrolls, grainsizes, bits, thd)

#define DEFINE_HISTBM_UNROLLS(grainsizes, bits, thd)                          \
    DEFINE_HISTBM_STRIPES(1, grainsizes, bits, thd)                           \
    DEFINE_HISTBM_STRIPES(2, grainsizes, bits, thd)                           \
    DEFINE_HISTBM_STRIPES(4, grainsizes, bits, thd)                           \
    DEFINE_HISTBM_STRIPES(8, grainsizes, bits, thd)                           \
    DEFINE_HISTBM_STRIPES(16, grainsizes, bits, thd)

#if BM_MULTITHREADED
#define DEFINE_HIST_BENCHMARKS(bits)                                          \
    DEFINE_HISTBM_UNROLLS(mt_grain_sizes, bits, mt)
#else
#define DEFINE_HIST_BENCHMARKS(bits)                                          \
    DEFINE_HISTBM_UNROLLS(st_grain_sizes, bits, st)
#endif

DEFINE_HIST_BENCHMARKS(BM_BITS)

} // namespace ihist::bench