/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include <ihist.hpp>

#include "parameterization.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ihist {

TEMPLATE_LIST_TEST_CASE("histogram counts are accumulated, not replaced", "",
                        test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    constexpr auto BITS = 8 * sizeof(T);
    constexpr auto NBINS = 1 << BITS;

    T const value = 42;
    constexpr std::size_t count = 10;
    std::vector<T> data(count, value);

    SECTION("1d pre-populated histogram gets counts added") {
        std::vector<std::uint32_t> hist(NBINS, 0);
        hist[value] = 100;

        std::vector<std::uint32_t> expected(NBINS, 0);
        expected[value] = 100 + count;

        constexpr auto *hist_func =
            traits::template hist_func<false, BITS, 0, 1, 0>;
        hist_func(data.data(), nullptr, count, hist.data(), 1);
        CHECK(hist == expected);
    }

    SECTION("2d pre-populated histogram gets counts added") {
        std::vector<std::uint32_t> hist(NBINS, 0);
        hist[value] = 50;

        std::vector<std::uint32_t> expected(NBINS, 0);
        expected[value] = 50 + count;

        constexpr auto *histxy_func =
            traits::template histxy_func<false, BITS, 0, 1, 0>;
        histxy_func(data.data(), nullptr, 2, 5, 5, 5, hist.data(), 1);
        CHECK(hist == expected);
    }

    SECTION("multiple calls accumulate") {
        std::vector<std::uint32_t> hist(NBINS, 0);

        constexpr auto *hist_func =
            traits::template hist_func<false, BITS, 0, 1, 0>;
        hist_func(data.data(), nullptr, count, hist.data(), 1);
        hist_func(data.data(), nullptr, count, hist.data(), 1);
        hist_func(data.data(), nullptr, count, hist.data(), 1);

        std::vector<std::uint32_t> expected(NBINS, 0);
        expected[value] = 3 * count;
        CHECK(hist == expected);
    }

    SECTION("pre-populated with multiple non-zero bins") {
        std::vector<std::uint32_t> hist(NBINS, 0);
        hist[0] = 10;
        hist[100] = 20;
        hist[value] = 30;
        hist[200] = 40;

        std::vector<std::uint32_t> expected(NBINS, 0);
        expected[0] = 10;
        expected[100] = 20;
        expected[value] = 30 + count;
        expected[200] = 40;

        constexpr auto *hist_func =
            traits::template hist_func<false, BITS, 0, 1, 0>;
        hist_func(data.data(), nullptr, count, hist.data(), 1);
        CHECK(hist == expected);
    }
}

TEMPLATE_LIST_TEST_CASE("multi-component histogram accumulation", "",
                        test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    constexpr auto BITS = 8 * sizeof(T);
    constexpr auto NBINS = 1 << BITS;

    T const value0 = 10;
    T const value1 = 20;
    T const value2 = 30;
    constexpr std::size_t count = 5;

    std::vector<T> data(4 * count);
    for (std::size_t i = 0; i < count; ++i) {
        data[4 * i + 0] = value0;
        data[4 * i + 1] = value1;
        data[4 * i + 3] = value2;
    }

    std::vector<std::uint32_t> hist(3 * NBINS, 0);
    hist[0 * NBINS + value0] = 100;
    hist[1 * NBINS + value1] = 200;
    hist[2 * NBINS + value2] = 300;

    std::vector<std::uint32_t> expected(3 * NBINS, 0);
    expected[0 * NBINS + value0] = 100 + count;
    expected[1 * NBINS + value1] = 200 + count;
    expected[2 * NBINS + value2] = 300 + count;

    constexpr auto *hist_func =
        traits::template hist_func<false, BITS, 0, 4, 0, 1, 3>;
    hist_func(data.data(), nullptr, count, hist.data(), 1);
    CHECK(hist == expected);
}

TEMPLATE_LIST_TEST_CASE("dynamic histogram accumulation", "",
                        dynamic_test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    constexpr auto BITS = 8 * sizeof(T);
    constexpr auto NBINS = 1 << BITS;

    constexpr std::size_t indices[] = {0, 1};
    T const value0 = 15;
    T const value1 = 25;
    constexpr std::size_t width = 4;
    constexpr std::size_t height = 3;

    std::vector<T> data(2 * width * height);
    for (std::size_t i = 0; i < width * height; ++i) {
        data[2 * i + 0] = value0;
        data[2 * i + 1] = value1;
    }

    std::vector<std::uint32_t> hist(2 * NBINS, 0);
    hist[0 * NBINS + value0] = 50;
    hist[1 * NBINS + value1] = 75;

    std::vector<std::uint32_t> expected(2 * NBINS, 0);
    expected[0 * NBINS + value0] = 50 + width * height;
    expected[1 * NBINS + value1] = 75 + width * height;

    traits::template histxy_dynamic<false, BITS, 0>(
        data.data(), nullptr, height, width, width, width, 2, 2, indices,
        hist.data());
    CHECK(hist == expected);
}

} // namespace ihist
