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
#include <numeric>
#include <vector>

namespace ihist {

// This mainly tests that the value-to-bin mapping is correct. For correct ROI
// and mask handling, see the constant-input tests.

TEMPLATE_LIST_TEST_CASE("all-values", "", test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    // Test two cases: (1) full bits of T and (2) half-width with quarter
    // shift. In both cases the input has full range of T.
    constexpr auto FULL_BITS = 8 * sizeof(T);
    constexpr auto FULL_NBINS = 1 << FULL_BITS;
    constexpr auto FULL_SHIFT = 0;
    constexpr auto HALF_BITS = FULL_BITS / 2;
    constexpr auto HALF_NBINS = 1 << HALF_BITS;
    constexpr auto HALF_SHIFT = FULL_BITS / 4;

    constexpr std::size_t width = 1 << HALF_BITS;
    constexpr std::size_t height = 1 << HALF_BITS;
    constexpr std::size_t size = width * height;

    std::vector<T> const data = [&] {
        std::vector<T> d(size);
        std::iota(d.begin(), d.end(), T(0));
        return d;
    }();
    std::vector<T> const data4 = [&] {
        std::vector<T> d4(4 * size);
        for (std::size_t i = 0; i < size; ++i) {
            d4[i * 4 + 0] = data[i];
            d4[i * 4 + 1] = data[i];
            d4[i * 4 + 3] = data[i];
        }
        return d4;
    }();

    SECTION("1d") {
        SECTION("mono") {
            SECTION("fullbits") {
                std::vector<std::uint32_t> hist(FULL_NBINS);
                std::vector<std::uint32_t> const expected(FULL_NBINS, 1);
                constexpr auto *hist_func =
                    traits::template hist_func<false, FULL_BITS, FULL_SHIFT, 1,
                                               0>;
                hist_func(data.data(), nullptr, size, hist.data(), 1);
                CHECK(hist == expected);
            }
            SECTION("halfbits") {
                std::vector<std::uint32_t> hist(HALF_NBINS);
                std::vector<std::uint32_t> const expected(HALF_NBINS,
                                                          1 << HALF_SHIFT);
                constexpr auto *hist_func =
                    traits::template hist_func<false, HALF_BITS, HALF_SHIFT, 1,
                                               0>;
                hist_func(data.data(), nullptr, size, hist.data(), 1);
                CHECK(hist == expected);
            }
        }
        SECTION("multi") {
            SECTION("fullbits") {
                std::vector<std::uint32_t> hist3(3 * FULL_NBINS);
                std::vector<std::uint32_t> const expected3(3 * FULL_NBINS, 1);
                constexpr auto *hist_func =
                    traits::template hist_func<false, FULL_BITS, FULL_SHIFT, 4,
                                               3, 0, 1>;
                hist_func(data4.data(), nullptr, size, hist3.data(), 1);
                CHECK(hist3 == expected3);
            }
            SECTION("halfbits") {
                std::vector<std::uint32_t> hist3(3 * HALF_NBINS);
                std::vector<std::uint32_t> const expected3(3 * HALF_NBINS,
                                                           1 << HALF_SHIFT);
                constexpr auto *hist_func =
                    traits::template hist_func<false, HALF_BITS, HALF_SHIFT, 4,
                                               3, 0, 1>;
                hist_func(data4.data(), nullptr, size, hist3.data(), 1);
                CHECK(hist3 == expected3);
            }
        }
    }

    SECTION("2d") {
        SECTION("mono") {
            SECTION("fullbits") {
                std::vector<std::uint32_t> hist(FULL_NBINS);
                std::vector<std::uint32_t> const expected(FULL_NBINS, 1);
                constexpr auto *histxy_func =
                    traits::template histxy_func<false, FULL_BITS, FULL_SHIFT,
                                                 1, 0>;
                histxy_func(data.data(), nullptr, height, width, width,
                            hist.data(), 1);
                CHECK(hist == expected);
            }
            SECTION("halfbits") {
                std::vector<std::uint32_t> hist(HALF_NBINS);
                std::vector<std::uint32_t> const expected(HALF_NBINS,
                                                          1 << HALF_SHIFT);
                constexpr auto *histxy_func =
                    traits::template histxy_func<false, HALF_BITS, HALF_SHIFT,
                                                 1, 0>;
                histxy_func(data.data(), nullptr, height, width, width,
                            hist.data(), 1);
                CHECK(hist == expected);
            }
        }
        SECTION("multi") {
            SECTION("fullbits") {
                std::vector<std::uint32_t> hist3(3 * FULL_NBINS);
                std::vector<std::uint32_t> const expected3(3 * FULL_NBINS, 1);
                constexpr auto *histxy_func =
                    traits::template histxy_func<false, FULL_BITS, FULL_SHIFT,
                                                 4, 3, 0, 1>;
                histxy_func(data4.data(), nullptr, height, width, width,
                            hist3.data(), 1);
                CHECK(hist3 == expected3);
            }
            SECTION("halfbits") {
                std::vector<std::uint32_t> hist3(3 * HALF_NBINS);
                std::vector<std::uint32_t> const expected3(3 * HALF_NBINS,
                                                           1 << HALF_SHIFT);
                constexpr auto *histxy_func =
                    traits::template histxy_func<false, HALF_BITS, HALF_SHIFT,
                                                 4, 3, 0, 1>;
                histxy_func(data4.data(), nullptr, height, width, width,
                            hist3.data(), 1);
                CHECK(hist3 == expected3);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE("dynamic-all-values", "", dynamic_test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    constexpr std::size_t indices[] = {0, 1};

    constexpr auto FULL_BITS = 8 * sizeof(T);
    constexpr auto FULL_NBINS = 1 << FULL_BITS;
    constexpr auto HALF_BITS = FULL_BITS / 2;
    constexpr auto HALF_NBINS = 1 << HALF_BITS;
    constexpr auto HALF_SHIFT = FULL_BITS / 4;

    constexpr std::size_t width = 1 << HALF_BITS;
    constexpr std::size_t height = 1 << HALF_BITS;
    constexpr std::size_t size = width * height;

    std::vector<T> const data = [&] {
        std::vector<T> d(2 * size);
        for (std::size_t i = 0; i < size; ++i) {
            d[2 * i + 0] = static_cast<T>(i);
            d[2 * i + 1] = static_cast<T>(i);
        }
        return d;
    }();

    SECTION("fullbits") {
        std::vector<std::uint32_t> hist(2 * FULL_NBINS);
        std::vector<std::uint32_t> const expected(2 * FULL_NBINS, 1);

        traits::template histxy_dynamic<false, FULL_BITS, 0>(
            data.data(), nullptr, height, width, width, 2, 2, indices,
            hist.data());
        CHECK(hist == expected);
    }

    SECTION("halfbits") {
        std::vector<std::uint32_t> hist(2 * HALF_NBINS);
        std::vector<std::uint32_t> const expected(2 * HALF_NBINS,
                                                  1 << HALF_SHIFT);

        traits::template histxy_dynamic<false, HALF_BITS, HALF_SHIFT>(
            data.data(), nullptr, height, width, width, 2, 2, indices,
            hist.data());
        CHECK(hist == expected);
    }
}

} // namespace ihist