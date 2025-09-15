/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include "ihist.hpp"
#include "ihist/ihist.h"

#include "benchmark_data.hpp"

#include <benchmark/benchmark.h>

#if IHIST_HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

namespace ihist::bench {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;

template <typename T>
using ihist_api_func = void(std::size_t, T const *, u8 const *, std::size_t,
                            std::size_t, std::size_t, std::size_t, std::size_t,
                            std::size_t, u32 *, bool);

template <typename T>
void bm_ihist_api(benchmark::State &state, ihist_api_func<T> *func,
                  std::size_t bits, std::size_t samples_per_pixel,
                  std::size_t n_components, bool masked, bool mt) {
    auto const width = state.range(0);
    auto const height = state.range(0);
    auto const size = width * height;
    auto const spread_frac = static_cast<float>(state.range(1)) / 100.0f;
    auto const data =
        generate_data<T>(bits, size * samples_per_pixel, spread_frac);
    auto const mask = generate_circle_mask(width, height);
    std::vector<u32> hist(n_components * (1 << bits));
    for ([[maybe_unused]] auto _ : state) {
        func(bits, data.data(), masked ? mask.data() : nullptr, width, height,
             0, 0, width, height, hist.data(), mt);
        benchmark::DoNotOptimize(hist.data());
    }
    state.SetBytesProcessed(static_cast<i64>(state.iterations()) * size *
                            samples_per_pixel * sizeof(T));
    state.counters["samples_per_second"] = benchmark::Counter(
        static_cast<i64>(state.iterations()) * size * n_components,
        benchmark::Counter::kIsRate);
    state.counters["pixels_per_second"] =
        benchmark::Counter(static_cast<i64>(state.iterations()) * size,
                           benchmark::Counter::kIsRate);
}

template <typename T>
using ihist_internal_func = void(T const *, u8 const *, std::size_t,
                                 std::size_t, std::size_t, std::size_t,
                                 std::size_t, std::size_t, u32 *, std::size_t);

template <typename T>
void bm_ihist_internal(benchmark::State &state, ihist_internal_func<T> *func,
                       std::size_t bits, std::size_t samples_per_pixel,
                       std::size_t n_components, bool masked) {
    auto const width = state.range(0);
    auto const height = state.range(0);
    auto const size = width * height;
    auto const spread_frac = static_cast<float>(state.range(1)) / 100.0f;
    auto const data =
        generate_data<T>(bits, size * samples_per_pixel, spread_frac);
    auto const mask = generate_circle_mask(width, height);
    std::vector<u32> hist(n_components * (1 << bits));
    for ([[maybe_unused]] auto _ : state) {
        func(data.data(), masked ? mask.data() : nullptr, width, height, 0, 0,
             width, height, hist.data(), 0);
        benchmark::DoNotOptimize(hist.data());
    }
    state.SetBytesProcessed(static_cast<i64>(state.iterations()) * size *
                            samples_per_pixel * sizeof(T));
    state.counters["samples_per_second"] = benchmark::Counter(
        static_cast<i64>(state.iterations()) * size * n_components,
        benchmark::Counter::kIsRate);
    state.counters["pixels_per_second"] =
        benchmark::Counter(static_cast<i64>(state.iterations()) * size,
                           benchmark::Counter::kIsRate);
}

#if IHIST_HAVE_OPENCV

template <typename T>
void opencv_histogram(T const *data, u8 const *mask, std::size_t width,
                      std::size_t height, std::size_t bits,
                      std::size_t samples_per_pixel, std::size_t n_components,
                      u32 *histogram) {
    auto const mat_type = [](int s) {
        if constexpr (std::is_same_v<T, u8>) {
            return CV_8UC(s);
        } else if constexpr (std::is_same_v<T, u16>) {
            return CV_16UC(s);
        } else {
            throw;
        }
    }(samples_per_pixel);
    cv::Mat const data_mat(height, width, mat_type, const_cast<T *>(data));
    cv::Mat mask_mat;
    if (mask != nullptr) {
        mask_mat = cv::Mat(height, width, CV_8UC1, const_cast<u8 *>(mask));
    }

    std::size_t const n_bins = 1 << bits;
    int const hist_size[] = {int(n_bins)};
    float const hist_range_0[] = {0, float(n_bins)};
    float const *hist_range[] = {hist_range_0};

    // OpenCV calcHist() does not perform multiple histograms in a single call;
    // the recommended method is to call for each component. Also, the produced
    // histogram is always float32 (for which OpenCV has good reason, but here
    // we convert back to u32 -- this overhead is probably small anyway).
    cv::Mat hist;
    for (std::size_t c = 0; c < n_components; ++c) {
        int const channels[] = {int(c)};
        cv::calcHist(&data_mat, 1, channels, mask_mat, hist, 1, hist_size,
                     hist_range);
        std::transform(hist.begin<float>(), hist.end<float>(),
                       std::next(histogram, c * n_bins),
                       [](float v) { return static_cast<u32>(v); });
    }
}

template <typename T>
void bm_opencv(benchmark::State &state, std::size_t bits,
               std::size_t samples_per_pixel, std::size_t n_components,
               bool masked, bool mt) {
    auto const save_nthreads = cv::getNumThreads();
    if (not mt) {
        cv::setNumThreads(1);
    }
    auto const width = state.range(0);
    auto const height = state.range(0);
    auto const size = width * height;
    auto const spread_frac = static_cast<float>(state.range(1)) / 100.0f;
    auto const data =
        generate_data<T>(bits, size * samples_per_pixel, spread_frac);
    auto const mask = generate_circle_mask(width, height);
    std::vector<u32> hist(n_components * (1 << bits));
    for ([[maybe_unused]] auto _ : state) {
        opencv_histogram(data.data(), masked ? mask.data() : nullptr, width,
                         height, bits, samples_per_pixel, n_components,
                         hist.data());
        benchmark::DoNotOptimize(hist.data());
    }
    state.SetBytesProcessed(static_cast<i64>(state.iterations()) * size *
                            samples_per_pixel * sizeof(T));
    state.counters["samples_per_second"] = benchmark::Counter(
        static_cast<i64>(state.iterations()) * size * n_components,
        benchmark::Counter::kIsRate);
    state.counters["pixels_per_second"] =
        benchmark::Counter(static_cast<i64>(state.iterations()) * size,
                           benchmark::Counter::kIsRate);
    cv::setNumThreads(save_nthreads);
}

#endif // IHIST_HAVE_OPENCV

std::vector const api_hist_8{
    ihist_hist8_mono_2d,
    ihist_hist8_abc_2d,
    ihist_hist8_abcx_2d,
};

std::vector const api_hist_16{
    ihist_hist16_mono_2d,
    ihist_hist16_abc_2d,
    ihist_hist16_abcx_2d,
};

std::vector const unoptimized_hist_8{
    std::vector{
        ihist::histxy_unoptimized_st<u8, false, 8, 0, 1, 0>,
        ihist::histxy_unoptimized_st<u8, false, 8, 0, 3, 0, 1, 2>,
        ihist::histxy_unoptimized_st<u8, false, 8, 0, 4, 0, 1, 2>,
    },
    std::vector{
        ihist::histxy_unoptimized_st<u8, true, 8, 0, 1, 0>,
        ihist::histxy_unoptimized_st<u8, true, 8, 0, 3, 0, 1, 2>,
        ihist::histxy_unoptimized_st<u8, true, 8, 0, 4, 0, 1, 2>,

    },
};

std::vector const unoptimized_hist_12{
    std::vector{
        ihist::histxy_unoptimized_st<u16, false, 12, 0, 1, 0>,
        ihist::histxy_unoptimized_st<u16, false, 12, 0, 3, 0, 1, 2>,
        ihist::histxy_unoptimized_st<u16, false, 12, 0, 4, 0, 1, 2>,
    },
    std::vector{
        ihist::histxy_unoptimized_st<u16, true, 12, 0, 1, 0>,
        ihist::histxy_unoptimized_st<u16, true, 12, 0, 3, 0, 1, 2>,
        ihist::histxy_unoptimized_st<u16, true, 12, 0, 4, 0, 1, 2>,

    },
};

std::vector const unoptimized_hist_16{
    std::vector{
        ihist::histxy_unoptimized_st<u16, false, 16, 0, 1, 0>,
        ihist::histxy_unoptimized_st<u16, false, 16, 0, 3, 0, 1, 2>,
        ihist::histxy_unoptimized_st<u16, false, 16, 0, 4, 0, 1, 2>,
    },
    std::vector{
        ihist::histxy_unoptimized_st<u16, true, 16, 0, 1, 0>,
        ihist::histxy_unoptimized_st<u16, true, 16, 0, 3, 0, 1, 2>,
        ihist::histxy_unoptimized_st<u16, true, 16, 0, 4, 0, 1, 2>,

    },
};

std::vector<i64> const data_sizes{512, 1024, 2048, 4096, 8192};

template <unsigned Bits> std::vector<i64> const spread_pcts{0, 1, 6, 25, 100};

} // namespace ihist::bench

auto main(int argc, char **argv) -> int {
    using namespace ihist::bench;

    auto register_benchmark = [](std::string const &name, auto lambda) {
        return benchmark::RegisterBenchmark(name.c_str(), lambda)
            ->MeasureProcessCPUTime()
            ->UseRealTime()
            ->ArgNames({"size", "spread"});
    };

    const std::vector<std::string> pixel_types{"mono", "abc", "abcx"};
    const std::vector<std::size_t> samples_per_pixel_values{1, 3, 4};
    const std::vector<std::size_t> component_counts{1, 3, 3};

    for (int mask = 0; mask < 2; ++mask) {
        std::string const mask_param = "mask:" + std::to_string(mask);

        for (std::size_t i = 0; i < pixel_types.size(); ++i) {
            std::string const pixel_type = pixel_types[i];
            std::size_t const samples_per_pixel = samples_per_pixel_values[i];
            std::size_t const n_components = component_counts[i];

            register_benchmark("unopt/" + pixel_type + "/bits:8/" + mask_param,
                               [=](benchmark::State &state) {
                                   bm_ihist_internal<u8>(
                                       state, unoptimized_hist_8[mask][i], 8,
                                       samples_per_pixel, n_components, mask);
                               })
                ->ArgsProduct({data_sizes, spread_pcts<8>});

            register_benchmark(
                "unopt/" + pixel_type + "/bits:12/" + mask_param,
                [=](benchmark::State &state) {
                    bm_ihist_internal<u16>(state, unoptimized_hist_12[mask][i],
                                           12, samples_per_pixel, n_components,
                                           mask);
                })
                ->ArgsProduct({data_sizes, spread_pcts<12>});

            register_benchmark(
                "unopt/" + pixel_type + "/bits:16/" + mask_param,
                [=](benchmark::State &state) {
                    bm_ihist_internal<u16>(state, unoptimized_hist_16[mask][i],
                                           16, samples_per_pixel, n_components,
                                           mask);
                })
                ->ArgsProduct({data_sizes, spread_pcts<16>});

            for (bool mt : {false, true}) {
                auto const *ihist_prefix = mt ? "ihist-mt/" : "ihist/";
                register_benchmark(
                    ihist_prefix + pixel_type + "/bits:8/" + mask_param,
                    [=](benchmark::State &state) {
                        bm_ihist_api<u8>(state, api_hist_8[i], 8,
                                         samples_per_pixel, n_components, mask,
                                         mt);
                    })
                    ->ArgsProduct({data_sizes, spread_pcts<8>});

                register_benchmark(
                    ihist_prefix + pixel_type + "/bits:12/" + mask_param,
                    [=](benchmark::State &state) {
                        bm_ihist_api<u16>(state, api_hist_16[i], 12,
                                          samples_per_pixel, n_components,
                                          mask, mt);
                    })
                    ->ArgsProduct({data_sizes, spread_pcts<12>});

                register_benchmark(
                    ihist_prefix + pixel_type + "/bits:16/" + mask_param,
                    [=](benchmark::State &state) {
                        bm_ihist_api<u16>(state, api_hist_16[i], 16,
                                          samples_per_pixel, n_components,
                                          mask, mt);
                    })
                    ->ArgsProduct({data_sizes, spread_pcts<8>});

#if IHIST_HAVE_OPENCV
                auto const *opencv_prefix = mt ? "opencv-mt/" : "opencv/";
                register_benchmark(
                    opencv_prefix + pixel_type + "/bits:8/" + mask_param,
                    [=](benchmark::State &state) {
                        bm_opencv<u8>(state, 8, samples_per_pixel,
                                      n_components, mask, mt);
                    })
                    ->ArgsProduct({data_sizes, spread_pcts<8>});

                register_benchmark(
                    opencv_prefix + pixel_type + "/bits:12/" + mask_param,
                    [=](benchmark::State &state) {
                        bm_opencv<u16>(state, 12, samples_per_pixel,
                                       n_components, mask, mt);
                    })
                    ->ArgsProduct({data_sizes, spread_pcts<12>});

                register_benchmark(
                    opencv_prefix + pixel_type + "/bits:16/" + mask_param,
                    [=](benchmark::State &state) {
                        bm_opencv<u16>(state, 16, samples_per_pixel,
                                       n_components, mask, mt);
                    })
                    ->ArgsProduct({data_sizes, spread_pcts<16>});
#endif
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
