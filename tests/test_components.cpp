/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include <ihist/ihist.h>

#include "ihist.hpp"

#include "test_data.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

namespace {

constexpr std::size_t width = 65;
constexpr std::size_t height = 63;
constexpr std::size_t roi_x = 7;
constexpr std::size_t roi_y = 5;
constexpr std::size_t roi_width = 33;
constexpr std::size_t roi_height = 29;
constexpr std::size_t size = width * height;

} // namespace

template <typename T, unsigned SampleBits> struct api_test_traits {
    using value_type = T;
    static constexpr unsigned sample_bits = SampleBits;
    static constexpr unsigned format_bits = 8 * sizeof(T);
    static constexpr std::size_t nbins = std::size_t{1} << SampleBits;

    static void call_api(T const *image, u8 const *mask, std::size_t h,
                         std::size_t w, std::size_t image_stride,
                         std::size_t mask_stride, std::size_t n_components,
                         std::size_t n_hist_components,
                         std::size_t const *component_indices, u32 *histogram,
                         bool parallel) {
        if constexpr (sizeof(T) == 1) {
            ihist_hist8_2d(SampleBits, image, mask, h, w, image_stride,
                           mask_stride, n_components, n_hist_components,
                           component_indices, histogram, parallel);
        } else {
            ihist_hist16_2d(SampleBits, image, mask, h, w, image_stride,
                            mask_stride, n_components, n_hist_components,
                            component_indices, histogram, parallel);
        }
    }
};

using api_test_traits_list =
    std::tuple<api_test_traits<u8, 8>, api_test_traits<u8, 5>,
               api_test_traits<u16, 16>, api_test_traits<u16, 15>,
               api_test_traits<u16, 12>, api_test_traits<u16, 11>>;

TEMPLATE_LIST_TEST_CASE("C API mono component", "", api_test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;
    constexpr auto sample_bits = traits::sample_bits;
    constexpr auto nbins = traits::nbins;

    auto const data = test_data<T, sample_bits>(size);
    auto const mask = test_data<u8, 1>(size);
    std::vector<u32> hist(nbins);
    std::vector<u32> ref(nbins);
    constexpr std::size_t indices[] = {0};

    bool const parallel = GENERATE(false, true);

    SECTION("nomask") {
        ihist::histxy_unoptimized_st<T, false, sample_bits, 0, 1, 0>(
            data.data() + roi_y * width + roi_x, nullptr, roi_height,
            roi_width, width, width, ref.data());
        traits::call_api(data.data() + roi_y * width + roi_x, nullptr,
                         roi_height, roi_width, width, width, 1, 1, indices,
                         hist.data(), parallel);
    }
    SECTION("mask") {
        ihist::histxy_unoptimized_st<T, true, sample_bits, 0, 1, 0>(
            data.data() + roi_y * width + roi_x,
            mask.data() + roi_y * width + roi_x, roi_height, roi_width, width,
            width, ref.data());
        traits::call_api(data.data() + roi_y * width + roi_x,
                         mask.data() + roi_y * width + roi_x, roi_height,
                         roi_width, width, width, 1, 1, indices, hist.data(),
                         parallel);
    }
    CHECK(hist == ref);
}

TEMPLATE_LIST_TEST_CASE("C API RGB (3 components, all selected)", "",
                        api_test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;
    constexpr auto sample_bits = traits::sample_bits;
    constexpr auto nbins = traits::nbins;

    auto const data = test_data<T, sample_bits>(3 * size);
    auto const mask = test_data<u8, 1>(size);
    std::vector<u32> hist(3 * nbins);
    std::vector<u32> ref(3 * nbins);
    constexpr std::size_t indices[] = {0, 1, 2};

    bool const parallel = GENERATE(false, true);

    SECTION("nomask") {
        ihist::histxy_unoptimized_st<T, false, sample_bits, 0, 3, 0, 1, 2>(
            data.data() + 3 * (roi_y * width + roi_x), nullptr, roi_height,
            roi_width, width, width, ref.data());
        traits::call_api(data.data() + 3 * (roi_y * width + roi_x), nullptr,
                         roi_height, roi_width, width, width, 3, 3, indices,
                         hist.data(), parallel);
    }
    SECTION("mask") {
        ihist::histxy_unoptimized_st<T, true, sample_bits, 0, 3, 0, 1, 2>(
            data.data() + 3 * (roi_y * width + roi_x),
            mask.data() + roi_y * width + roi_x, roi_height, roi_width, width,
            width, ref.data());
        traits::call_api(data.data() + 3 * (roi_y * width + roi_x),
                         mask.data() + roi_y * width + roi_x, roi_height,
                         roi_width, width, width, 3, 3, indices, hist.data(),
                         parallel);
    }
    CHECK(hist == ref);
}

TEMPLATE_LIST_TEST_CASE("C API RGBX (4 components, skip last)", "",
                        api_test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;
    constexpr auto sample_bits = traits::sample_bits;
    constexpr auto nbins = traits::nbins;

    auto const data = test_data<T, sample_bits>(4 * size);
    auto const mask = test_data<u8, 1>(size);
    std::vector<u32> hist(3 * nbins);
    std::vector<u32> ref(3 * nbins);
    constexpr std::size_t indices[] = {0, 1, 2};

    bool const parallel = GENERATE(false, true);

    SECTION("nomask") {
        ihist::histxy_unoptimized_st<T, false, sample_bits, 0, 4, 0, 1, 2>(
            data.data() + 4 * (roi_y * width + roi_x), nullptr, roi_height,
            roi_width, width, width, ref.data());
        traits::call_api(data.data() + 4 * (roi_y * width + roi_x), nullptr,
                         roi_height, roi_width, width, width, 4, 3, indices,
                         hist.data(), parallel);
    }
    SECTION("mask") {
        ihist::histxy_unoptimized_st<T, true, sample_bits, 0, 4, 0, 1, 2>(
            data.data() + 4 * (roi_y * width + roi_x),
            mask.data() + roi_y * width + roi_x, roi_height, roi_width, width,
            width, ref.data());
        traits::call_api(data.data() + 4 * (roi_y * width + roi_x),
                         mask.data() + roi_y * width + roi_x, roi_height,
                         roi_width, width, width, 4, 3, indices, hist.data(),
                         parallel);
    }
    CHECK(hist == ref);
}

TEMPLATE_LIST_TEST_CASE("C API XRGB (4 components, skip first)", "",
                        api_test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;
    constexpr auto sample_bits = traits::sample_bits;
    constexpr auto nbins = traits::nbins;

    auto const data = test_data<T, sample_bits>(4 * size);
    auto const mask = test_data<u8, 1>(size);
    std::vector<u32> hist(3 * nbins);
    std::vector<u32> ref(3 * nbins);
    constexpr std::size_t indices[] = {1, 2, 3};

    bool const parallel = GENERATE(false, true);

    SECTION("nomask") {
        ihist::histxy_unoptimized_st<T, false, sample_bits, 0, 4, 1, 2, 3>(
            data.data() + 4 * (roi_y * width + roi_x), nullptr, roi_height,
            roi_width, width, width, ref.data());
        traits::call_api(data.data() + 4 * (roi_y * width + roi_x), nullptr,
                         roi_height, roi_width, width, width, 4, 3, indices,
                         hist.data(), parallel);
    }
    SECTION("mask") {
        ihist::histxy_unoptimized_st<T, true, sample_bits, 0, 4, 1, 2, 3>(
            data.data() + 4 * (roi_y * width + roi_x),
            mask.data() + roi_y * width + roi_x, roi_height, roi_width, width,
            width, ref.data());
        traits::call_api(data.data() + 4 * (roi_y * width + roi_x),
                         mask.data() + roi_y * width + roi_x, roi_height,
                         roi_width, width, width, 4, 3, indices, hist.data(),
                         parallel);
    }
    CHECK(hist == ref);
}

TEMPLATE_LIST_TEST_CASE("C API dual component", "", api_test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;
    constexpr auto sample_bits = traits::sample_bits;
    constexpr auto nbins = traits::nbins;

    auto const data = test_data<T, sample_bits>(2 * size);
    auto const mask = test_data<u8, 1>(size);
    std::vector<u32> hist(2 * nbins);
    std::vector<u32> ref(2 * nbins);
    constexpr std::size_t indices[] = {0, 1};

    bool const parallel = GENERATE(false, true);

    SECTION("nomask") {
        ihist::histxy_unoptimized_st<T, false, sample_bits, 0, 2, 0, 1>(
            data.data() + 2 * (roi_y * width + roi_x), nullptr, roi_height,
            roi_width, width, width, ref.data());
        traits::call_api(data.data() + 2 * (roi_y * width + roi_x), nullptr,
                         roi_height, roi_width, width, width, 2, 2, indices,
                         hist.data(), parallel);
    }
    SECTION("mask") {
        ihist::histxy_unoptimized_st<T, true, sample_bits, 0, 2, 0, 1>(
            data.data() + 2 * (roi_y * width + roi_x),
            mask.data() + roi_y * width + roi_x, roi_height, roi_width, width,
            width, ref.data());
        traits::call_api(data.data() + 2 * (roi_y * width + roi_x),
                         mask.data() + roi_y * width + roi_x, roi_height,
                         roi_width, width, width, 2, 2, indices, hist.data(),
                         parallel);
    }
    CHECK(hist == ref);
}

TEMPLATE_LIST_TEST_CASE("C API non-contiguous component indices", "",
                        api_test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;
    constexpr auto sample_bits = traits::sample_bits;
    constexpr auto nbins = traits::nbins;

    auto const data = test_data<T, sample_bits>(4 * size);
    auto const mask = test_data<u8, 1>(size);
    std::vector<u32> hist(2 * nbins);
    std::vector<u32> ref(2 * nbins);
    constexpr std::size_t indices[] = {0, 3};

    bool const parallel = GENERATE(false, true);

    SECTION("nomask") {
        ihist::histxy_unoptimized_st<T, false, sample_bits, 0, 4, 0, 3>(
            data.data() + 4 * (roi_y * width + roi_x), nullptr, roi_height,
            roi_width, width, width, ref.data());
        traits::call_api(data.data() + 4 * (roi_y * width + roi_x), nullptr,
                         roi_height, roi_width, width, width, 4, 2, indices,
                         hist.data(), parallel);
    }
    SECTION("mask") {
        ihist::histxy_unoptimized_st<T, true, sample_bits, 0, 4, 0, 3>(
            data.data() + 4 * (roi_y * width + roi_x),
            mask.data() + roi_y * width + roi_x, roi_height, roi_width, width,
            width, ref.data());
        traits::call_api(data.data() + 4 * (roi_y * width + roi_x),
                         mask.data() + roi_y * width + roi_x, roi_height,
                         roi_width, width, width, 4, 2, indices, hist.data(),
                         parallel);
    }
    CHECK(hist == ref);
}
