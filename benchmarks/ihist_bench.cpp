/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include "ihist.hpp"

#include "benchmark_data.hpp"
#include "tmpl_instantiations.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ihist::bench {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;

enum class input_dim {
    one_d,
    two_d,
};

auto input_dim_name(input_dim d) -> std::string {
    switch (d) {
    case input_dim::one_d:
        return "1d";
    case input_dim::two_d:
        return "2d";
    }
    throw;
}

enum class pixel_type {
    mono,
    abc,
    abcx,
};

auto pixel_type_name(pixel_type t) -> std::string {
    switch (t) {
    case pixel_type::mono:
        return "mono";
    case pixel_type::abc:
        return "abc";
    case pixel_type::abcx:
        return "abcx";
    }
    throw;
}

// Return stride, n_components
constexpr auto pixel_type_attrs(pixel_type t)
    -> std::pair<std::size_t, std::size_t> {
    switch (t) {
    case pixel_type::mono:
        return {1, 1};
    case pixel_type::abc:
        return {3, 3};
    case pixel_type::abcx:
        return {4, 3};
    }
    throw;
}

template <typename T>
using hist_func = void(T const *, u8 const *, std::size_t, u32 *, std::size_t);

template <typename T>
using histxy_func = void(T const *, u8 const *, std::size_t, std::size_t,
                         std::size_t, std::size_t, std::size_t, std::size_t,
                         u32 *, std::size_t);

template <typename T>
void bm_hist(benchmark::State &state, hist_func<T> *func, std::size_t bits,
             pixel_type ptype) {
    auto const [stride, n_components] = pixel_type_attrs(ptype);
    auto const width = state.range(0);
    auto const height = state.range(0);
    auto const size = width * height;
    auto const spread_frac = static_cast<float>(state.range(1)) / 100.0f;
    auto const grain_size = static_cast<std::size_t>(state.range(2));
    auto const data = generate_data<T>(bits, size * stride, spread_frac);
    auto const mask = generate_circle_mask(width, height);
    std::vector<u32> hist(n_components * (1 << bits));
    for ([[maybe_unused]] auto _ : state) {
        std::fill(hist.begin(), hist.end(), 0);
        func(data.data(), mask.data(), size, hist.data(), grain_size);
        benchmark::DoNotOptimize(hist);
    }
    state.SetBytesProcessed(static_cast<i64>(state.iterations()) * size *
                            stride * sizeof(T));
    state.counters["samples_per_second"] = benchmark::Counter(
        static_cast<i64>(state.iterations()) * size * n_components,
        benchmark::Counter::kIsRate);
    state.counters["pixels_per_second"] =
        benchmark::Counter(static_cast<i64>(state.iterations()) * size,
                           benchmark::Counter::kIsRate);
}

template <typename T>
void bm_histxy(benchmark::State &state, histxy_func<T> *func, std::size_t bits,
               pixel_type ptype) {
    auto const [stride, n_components] = pixel_type_attrs(ptype);
    auto const width = state.range(0);
    auto const height = state.range(0);
    auto const size = width * height;
    auto const spread_frac = static_cast<float>(state.range(1)) / 100.0f;
    auto const grain_size = static_cast<std::size_t>(state.range(2));
    // For now, ROI is full image.
    auto const roi_size = width * height;
    auto const data = generate_data<T>(bits, size * stride, spread_frac);
    auto const mask = generate_circle_mask(width, height);
    std::vector<u32> hist(n_components * (1 << bits));
    for ([[maybe_unused]] auto _ : state) {
        std::fill(hist.begin(), hist.end(), 0);
        func(data.data(), mask.data(), width, height, 0, 0, width, height,
             hist.data(), grain_size);
        benchmark::DoNotOptimize(hist);
    }
    state.SetBytesProcessed(static_cast<i64>(state.iterations()) * roi_size *
                            stride * sizeof(T));
    state.counters["samples_per_second"] = benchmark::Counter(
        static_cast<i64>(state.iterations()) * roi_size * n_components,
        benchmark::Counter::kIsRate);
    state.counters["pixels_per_second"] =
        benchmark::Counter(static_cast<i64>(state.iterations()) * roi_size,
                           benchmark::Counter::kIsRate);
}

// The spread of the data affects performance: if narrow, a simple
// implementation will be bound by store-to-load forwarding latencyes due to
// incrementing a bin on the same cache line in close succession. Striped
// implementations may slow down for a wide distribution due to the larger
// working set size.
// 6% and 25% are useful for comparing 16-bit performance with 12/14-bit
// performance.
std::vector<i64> const spread_pcts{0, 1, 6, 25, 100};

// For single-threaded, performance starts to drop when the data no longer fits
// in the last-level cache, but that is not an effect we are particularly
// interested in (because there is not much we can do to prevent it). For
// multi-threaded, it is hard to reach a performnce plateau, even with hundreds
// of megapixels. 16 Mi (1 << 24) is probably a reasonable compromize if
// looking at a single size (and the image size of large-ish CMOS chips).
// data_sizes is the width and height of the data.
std::vector<i64> const data_sizes{1 << 12};

auto grain_sizes(bool mt, std::size_t hist_size) -> std::vector<i64> {
    if (mt) {
        // We scale the grain size (roughly: 2x the minimum task size to run)
        // according to the histogram size, because the primary factor to
        // consider is the overhead of the per-thread stripe reduction, which
        // is duplicated for each task.
        std::vector<i64> ret(6);
        std::iota(ret.begin(), ret.end(), 0);
        std::transform(ret.begin(), ret.end(), ret.begin(), [&](i64 i) {
            auto const shift = i - 2;
            if (shift < 0)
                return i64(hist_size >> -shift);
            else
                return i64(hist_size << shift);
        });
        return ret;
    } else {
        return {0};
    }
}

auto bench_name(pixel_type ptype, std::size_t bits, input_dim dim, bool mask,
                bool mt, std::size_t stripes, std::size_t unrolls)
    -> std::string {
    return pixel_type_name(ptype) + "/bits:" + std::to_string(bits) +
           "/input:" + input_dim_name(dim) + "/mask:" + (mask ? '1' : '0') +
           "/mt:" + (mt ? '1' : '0') + "/stripes:" + std::to_string(stripes) +
           "/unrolls:" + std::to_string(unrolls);
}

// Tag types for 'Dim' parameter.
struct input_dim_tag_1d {};
struct input_dim_tag_2d {};

template <typename Dim, typename T, std::size_t Bits, pixel_type PixelType,
          bool Mask, bool MT, std::size_t Stripes, std::size_t Unrolls>
constexpr auto static_hist_func() {
    if constexpr (std::is_same_v<Dim, input_dim_tag_1d>) {
        if constexpr (MT) {
            if constexpr (PixelType == pixel_type::mono) {
                return hist_striped_mt<tuning<Stripes, Unrolls>, T, Mask, Bits,
                                       0, 1, 0>;
            } else if constexpr (PixelType == pixel_type::abc) {
                return hist_striped_mt<tuning<Stripes, Unrolls>, T, Mask, Bits,
                                       0, 3, 0, 1, 2>;
            } else if constexpr (PixelType == pixel_type::abcx) {
                return hist_striped_mt<tuning<Stripes, Unrolls>, T, Mask, Bits,
                                       0, 4, 0, 1, 2>;
            }
        } else {
            if constexpr (PixelType == pixel_type::mono) {
                return hist_striped_st<tuning<Stripes, Unrolls>, T, Mask, Bits,
                                       0, 1, 0>;
            } else if constexpr (PixelType == pixel_type::abc) {
                return hist_striped_st<tuning<Stripes, Unrolls>, T, Mask, Bits,
                                       0, 3, 0, 1, 2>;
            } else if constexpr (PixelType == pixel_type::abcx) {
                return hist_striped_st<tuning<Stripes, Unrolls>, T, Mask, Bits,
                                       0, 4, 0, 1, 2>;
            }
        }
    } else if constexpr (std::is_same_v<Dim, input_dim_tag_2d>) {
        if constexpr (MT) {
            if constexpr (PixelType == pixel_type::mono) {
                return histxy_striped_mt<tuning<Stripes, Unrolls>, T, Mask,
                                         Bits, 0, 1, 0>;
            } else if constexpr (PixelType == pixel_type::abc) {
                return histxy_striped_mt<tuning<Stripes, Unrolls>, T, Mask,
                                         Bits, 0, 3, 0, 1, 2>;
            } else if constexpr (PixelType == pixel_type::abcx) {
                return histxy_striped_mt<tuning<Stripes, Unrolls>, T, Mask,
                                         Bits, 0, 4, 0, 1, 2>;
            }
        } else {
            if constexpr (PixelType == pixel_type::mono) {
                return histxy_striped_st<tuning<Stripes, Unrolls>, T, Mask,
                                         Bits, 0, 1, 0>;
            } else if constexpr (PixelType == pixel_type::abc) {
                return histxy_striped_st<tuning<Stripes, Unrolls>, T, Mask,
                                         Bits, 0, 3, 0, 1, 2>;
            } else if constexpr (PixelType == pixel_type::abcx) {
                return histxy_striped_st<tuning<Stripes, Unrolls>, T, Mask,
                                         Bits, 0, 4, 0, 1, 2>;
            }
        }
    }
    throw;
}

template <typename Dim, typename T, std::size_t Bits, pixel_type PixelType,
          bool Mask, bool MT, std::size_t Stripes>
constexpr auto dyn_hist_func_stripes(std::size_t unrolls) {
    switch (unrolls) {
    case 1:
        return static_hist_func<Dim, T, Bits, PixelType, Mask, MT, Stripes,
                                1>();
    case 2:
        return static_hist_func<Dim, T, Bits, PixelType, Mask, MT, Stripes,
                                2>();
    case 4:
        return static_hist_func<Dim, T, Bits, PixelType, Mask, MT, Stripes,
                                4>();
    case 8:
        return static_hist_func<Dim, T, Bits, PixelType, Mask, MT, Stripes,
                                8>();
    case 16:
        return static_hist_func<Dim, T, Bits, PixelType, Mask, MT, Stripes,
                                16>();
    }
    throw;
}

template <typename Dim, typename T, std::size_t Bits, pixel_type PixelType,
          bool Mask, bool MT>
constexpr auto dyn_hist_func_mt(std::size_t stripes, std::size_t unrolls) {
    switch (stripes) {
    case 1:
        return dyn_hist_func_stripes<Dim, T, Bits, PixelType, Mask, MT, 1>(
            unrolls);
    case 2:
        return dyn_hist_func_stripes<Dim, T, Bits, PixelType, Mask, MT, 2>(
            unrolls);
    case 4:
        return dyn_hist_func_stripes<Dim, T, Bits, PixelType, Mask, MT, 4>(
            unrolls);
    case 8:
        return dyn_hist_func_stripes<Dim, T, Bits, PixelType, Mask, MT, 8>(
            unrolls);
    case 16:
        return dyn_hist_func_stripes<Dim, T, Bits, PixelType, Mask, MT, 16>(
            unrolls);
    }
    throw;
}

template <typename Dim, typename T, std::size_t Bits, pixel_type PixelType,
          bool Mask>
constexpr auto dyn_hist_func_mask(bool mt, std::size_t stripes,
                                  std::size_t unrolls) {
    if (mt) {
        return dyn_hist_func_mt<Dim, T, Bits, PixelType, Mask, true>(stripes,
                                                                     unrolls);
    } else {
        return dyn_hist_func_mt<Dim, T, Bits, PixelType, Mask, false>(stripes,
                                                                      unrolls);
    }
}

template <typename Dim, typename T, std::size_t Bits, pixel_type PixelType>
constexpr auto dyn_hist_func_pixel_type(bool mask, bool mt,
                                        std::size_t stripes,
                                        std::size_t unrolls) {
    if (mask) {
        return dyn_hist_func_mask<Dim, T, Bits, PixelType, true>(mt, stripes,
                                                                 unrolls);
    } else {
        return dyn_hist_func_mask<Dim, T, Bits, PixelType, false>(mt, stripes,
                                                                  unrolls);
    }
}

template <typename Dim, typename T, std::size_t Bits>
constexpr auto dyn_hist_func_bits(pixel_type ptype, bool mask, bool mt,
                                  std::size_t stripes, std::size_t unrolls) {
    switch (ptype) {
    case pixel_type::mono:
        return dyn_hist_func_pixel_type<Dim, T, Bits, pixel_type::mono>(
            mask, mt, stripes, unrolls);
    case pixel_type::abc:
        return dyn_hist_func_pixel_type<Dim, T, Bits, pixel_type::abc>(
            mask, mt, stripes, unrolls);
    case pixel_type::abcx:
        return dyn_hist_func_pixel_type<Dim, T, Bits, pixel_type::abcx>(
            mask, mt, stripes, unrolls);
    }
    throw;
}

template <typename Dim, typename T>
constexpr auto dyn_hist_func(std::size_t bits, pixel_type ptype, bool mask,
                             bool mt, std::size_t stripes,
                             std::size_t unrolls) {
    if constexpr (std::is_same_v<T, u8>) {
        switch (bits) {
        case 8:
            return dyn_hist_func_bits<Dim, u8, 8>(ptype, mask, mt, stripes,
                                                  unrolls);
        }
    } else if constexpr (std::is_same_v<T, u16>) {
        switch (bits) {
        case 10:
            return dyn_hist_func_bits<Dim, u16, 10>(ptype, mask, mt, stripes,
                                                    unrolls);
        case 12:
            return dyn_hist_func_bits<Dim, u16, 12>(ptype, mask, mt, stripes,
                                                    unrolls);
        case 14:
            return dyn_hist_func_bits<Dim, u16, 14>(ptype, mask, mt, stripes,
                                                    unrolls);
        case 16:
            return dyn_hist_func_bits<Dim, u16, 16>(ptype, mask, mt, stripes,
                                                    unrolls);
        }
    }
    throw;
}

} // namespace ihist::bench

auto main(int argc, char **argv) -> int {
    using namespace ihist::bench;

    auto register_benchmark = [](std::string const &name, auto lambda) {
        return benchmark::RegisterBenchmark(name.c_str(), lambda)
            ->MeasureProcessCPUTime()
            ->UseRealTime()
            ->ArgNames({"size", "spread", "grainsize"});
    };

    std::vector<std::size_t> const u8_bits{8};
    std::vector<std::size_t> const u16_bits{10, 12, 14, 16};
    std::vector<pixel_type> const pixel_types{
        pixel_type::mono, pixel_type::abc, pixel_type::abcx};
    constexpr std::array<bool, 2> masking{false, true};
    constexpr std::array<bool, 2> st_mt{false, true};
    std::vector<std::size_t> const stripes{1, 2, 4, 8, 16};
    std::vector<std::size_t> const unrolls{1, 2, 4, 8, 16};
    for (auto ptype : pixel_types) {
        for (auto mask : masking) {
            for (auto mt : st_mt) {
                for (auto s : stripes) {
                    for (auto u : unrolls) {
                        if (ptype != pixel_type::mono && (s > 4 || u > 4)) {
                            continue;
                        }
                        for (auto bits : u8_bits) {
                            std::size_t const hist_size =
                                (1uLL << bits) *
                                (ptype == pixel_type::mono ? 1 : 3);
                            register_benchmark(
                                bench_name(ptype, bits, input_dim::one_d, mask,
                                           mt, s, u),
                                [=](benchmark::State &state) {
                                    bm_hist<u8>(
                                        state,
                                        dyn_hist_func<input_dim_tag_1d, u8>(
                                            bits, ptype, mask, mt, s, u),
                                        bits, ptype);
                                })
                                ->ArgsProduct({data_sizes, spread_pcts,
                                               grain_sizes(mt, hist_size)});
                            register_benchmark(
                                bench_name(ptype, bits, input_dim::two_d, mask,
                                           mt, s, u),
                                [=](benchmark::State &state) {
                                    bm_histxy<u8>(
                                        state,
                                        dyn_hist_func<input_dim_tag_2d, u8>(
                                            bits, ptype, mask, mt, s, u),
                                        bits, ptype);
                                })
                                ->ArgsProduct({data_sizes, spread_pcts,
                                               grain_sizes(mt, hist_size)});
                        }
                        for (auto bits : u16_bits) {
                            std::size_t const hist_size =
                                (1uLL << bits) *
                                (ptype == pixel_type::mono ? 1 : 3);
                            register_benchmark(
                                bench_name(ptype, bits, input_dim::one_d, mask,
                                           mt, s, u),
                                [=](benchmark::State &state) {
                                    bm_hist<u16>(
                                        state,
                                        dyn_hist_func<input_dim_tag_1d, u16>(
                                            bits, ptype, mask, mt, s, u),
                                        bits, ptype);
                                })
                                ->ArgsProduct({data_sizes, spread_pcts,
                                               grain_sizes(mt, hist_size)});
                            register_benchmark(
                                bench_name(ptype, bits, input_dim::two_d, mask,
                                           mt, s, u),
                                [=](benchmark::State &state) {
                                    bm_histxy<u16>(
                                        state,
                                        dyn_hist_func<input_dim_tag_2d, u16>(
                                            bits, ptype, mask, mt, s, u),
                                        bits, ptype);
                                })
                                ->ArgsProduct({data_sizes, spread_pcts,
                                               grain_sizes(mt, hist_size)});
                        }
                    }
                }
            }
        }
    }

    using namespace benchmark;
    Initialize(&argc, argv);
    if (ReportUnrecognizedArguments(argc, argv))
        return 1;
    RunSpecifiedBenchmarks();
    Shutdown();

    return 0;
}