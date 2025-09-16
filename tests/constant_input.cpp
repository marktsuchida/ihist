/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include <ihist.hpp>

#include "parameterization.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ihist {

// This mainly tests that the correct parts of the input are histogrammed. For
// correct value-to-bin mapping, see the all-values tests.

TEMPLATE_LIST_TEST_CASE("constant-input", "", test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    // Test two cases: (1) full bits of T and (2) half-width with quarter
    // shift.
    constexpr auto FULL_BITS = 8 * sizeof(T);
    constexpr auto FULL_NBINS = 1 << FULL_BITS;
    constexpr auto FULL_SHIFT = 0;
    constexpr auto HALF_BITS = FULL_BITS / 2;
    constexpr auto HALF_NBINS = 1 << HALF_BITS;
    constexpr auto HALF_SHIFT = FULL_BITS / 4;

    std::size_t const width = GENERATE(1, 3, 100);
    std::size_t const height = GENERATE(1, 7);
    // Generate rectangular ROI, leaving out empty cases (tested elsewhere).
    auto const quad_x = GENERATE_COPY(
        filter([=](auto x) { return x < width; },
               values<std::size_t>({0, 1, width > 2 ? width - 1 : 9999})));
    auto const quad_y = GENERATE_COPY(
        filter([=](auto y) { return y < height; },
               values<std::size_t>({0, 1, height > 2 ? height - 1 : 9999})));
    auto const quad_width = GENERATE_COPY(
        filter([=](auto w) { return w <= width - quad_x; },
               values<std::size_t>(
                   {1, width - quad_x > 2 ? width - quad_x - 1 : 9999})));
    auto const quad_height = GENERATE_COPY(
        filter([=](auto h) { return h <= height - quad_y; },
               values<std::size_t>(
                   {1, height - quad_y > 2 ? height - quad_y - 1 : 9999})));
    CAPTURE(width, height, quad_x, quad_y, quad_width, quad_height);

    std::size_t const size = width * height;
    std::size_t const quad_size = quad_width * quad_height;

    // We have separate all-value tests, so only test one value here.
    T const value_in_roi = 1;
    T const value_not_in_roi = 63;

    std::vector<T> const full_data(size, value_in_roi);
    std::vector<T> const quad_data = [&] {
        std::vector<T> data(size, value_not_in_roi);
        for (std::size_t y = quad_y; y < quad_y + quad_height; ++y) {
            std::fill_n(data.begin() + y * width + quad_x, quad_width,
                        value_in_roi);
        }
        return data;
    }();
    std::vector<T> const full4_data = [&] {
        std::vector<T> data(4 * size, value_not_in_roi);
        for (std::size_t i = 0; i < data.size(); i += 4) {
            data[i + 0] = value_in_roi;
            data[i + 1] = value_in_roi;
            data[i + 3] = value_in_roi;
        }
        return data;
    }();
    std::vector<T> const quad4_data = [&] {
        std::vector<T> data(4 * size, value_not_in_roi);
        for (std::size_t i = 0; i < data.size(); i += 4) {
            auto const x = i / 4 % width;
            auto const y = i / 4 / width;
            if (x >= quad_x && x < quad_x + quad_width && y >= quad_y &&
                y < quad_y + quad_height) {
                data[i + 0] = value_in_roi;
                data[i + 1] = value_in_roi;
                data[i + 3] = value_in_roi;
            }
        }
        return data;
    }();
    std::vector<std::uint8_t> const full_mask(size, 1);
    std::vector<std::uint8_t> const quad_mask = [&] {
        std::vector<std::uint8_t> mask(size);
        for (std::size_t y = quad_y; y < quad_y + quad_height; ++y) {
            std::fill_n(mask.begin() + y * width + quad_x, quad_width, 1);
        }
        return mask;
    }();

    SECTION("1d") {
        SECTION("mono") {
            SECTION("fullbits") {
                std::vector<std::uint32_t> hist(FULL_NBINS);
                SECTION("nomask") {
                    auto const expected = [&] {
                        std::vector<std::uint32_t> exp(FULL_NBINS);
                        exp[value_in_roi >> FULL_SHIFT] = size;
                        return exp;
                    }();
                    constexpr auto *hist_func =
                        traits::template hist_func<false, FULL_BITS,
                                                   FULL_SHIFT, 1, 0>;
                    hist_func(full_data.data(), nullptr, size, hist.data(), 1);
                    CHECK(hist == expected);
                }
                SECTION("mask") {
                    auto const expected = [&] {
                        std::vector<std::uint32_t> exp(FULL_NBINS);
                        exp[value_in_roi >> FULL_SHIFT] = quad_size;
                        return exp;
                    }();
                    constexpr auto *hist_func =
                        traits::template hist_func<true, FULL_BITS, FULL_SHIFT,
                                                   1, 0>;
                    hist_func(quad_data.data(), quad_mask.data(), size,
                              hist.data(), 1);
                    CHECK(hist == expected);
                }
            }
            SECTION("halfbits") {
                std::vector<std::uint32_t> hist(HALF_NBINS);
                SECTION("nomask") {
                    auto const expected = [&] {
                        std::vector<std::uint32_t> exp(HALF_NBINS);
                        exp[value_in_roi >> HALF_SHIFT] = size;
                        return exp;
                    }();
                    constexpr auto *hist_func =
                        traits::template hist_func<false, HALF_BITS,
                                                   HALF_SHIFT, 1, 0>;
                    hist_func(full_data.data(), nullptr, size, hist.data(), 1);
                    CHECK(hist == expected);
                }
                SECTION("mask") {
                    auto const expected = [&] {
                        std::vector<std::uint32_t> exp(HALF_NBINS);
                        exp[value_in_roi >> HALF_SHIFT] = quad_size;
                        return exp;
                    }();
                    constexpr auto *hist_func =
                        traits::template hist_func<true, HALF_BITS, HALF_SHIFT,
                                                   1, 0>;
                    hist_func(quad_data.data(), quad_mask.data(), size,
                              hist.data(), 1);
                    CHECK(hist == expected);
                }
            }
        }
        SECTION("multi") {
            SECTION("fullbits") {
                std::vector<std::uint32_t> hist3(3 * FULL_NBINS);
                SECTION("nomask") {
                    auto const expected3 = [&] {
                        std::vector<std::uint32_t> exp(3 * FULL_NBINS);
                        auto const bin = value_in_roi >> FULL_SHIFT;
                        exp[0 * FULL_NBINS + bin] = size;
                        exp[1 * FULL_NBINS + bin] = size;
                        exp[2 * FULL_NBINS + bin] = size;
                        return exp;
                    }();
                    constexpr auto *hist_func =
                        traits::template hist_func<false, FULL_BITS,
                                                   FULL_SHIFT, 4, 3, 0, 1>;
                    hist_func(full4_data.data(), nullptr, size, hist3.data(),
                              1);
                    CHECK(hist3 == expected3);
                }
                SECTION("mask") {
                    auto const expected3 = [&] {
                        std::vector<std::uint32_t> exp(3 * FULL_NBINS);
                        auto const bin = value_in_roi >> FULL_SHIFT;
                        exp[0 * FULL_NBINS + bin] = quad_size;
                        exp[1 * FULL_NBINS + bin] = quad_size;
                        exp[2 * FULL_NBINS + bin] = quad_size;
                        return exp;
                    }();
                    constexpr auto *hist_func =
                        traits::template hist_func<true, FULL_BITS, FULL_SHIFT,
                                                   4, 3, 0, 1>;
                    hist_func(quad4_data.data(), quad_mask.data(), size,
                              hist3.data(), 1);
                    CHECK(hist3 == expected3);
                }
            }
            SECTION("halfbits") {
                std::vector<std::uint32_t> hist3(3 * HALF_NBINS);
                SECTION("nomask") {
                    auto const expected3 = [&] {
                        std::vector<std::uint32_t> exp(3 * HALF_NBINS);
                        auto const bin = value_in_roi >> HALF_SHIFT;
                        exp[0 * HALF_NBINS + bin] = size;
                        exp[1 * HALF_NBINS + bin] = size;
                        exp[2 * HALF_NBINS + bin] = size;
                        return exp;
                    }();
                    constexpr auto *hist_func =
                        traits::template hist_func<false, HALF_BITS,
                                                   HALF_SHIFT, 4, 3, 0, 1>;
                    hist_func(full4_data.data(), nullptr, size, hist3.data(),
                              1);
                    CHECK(hist3 == expected3);
                }
                SECTION("mask") {
                    auto const expected3 = [&] {
                        std::vector<std::uint32_t> exp(3 * HALF_NBINS);
                        auto const bin = value_in_roi >> HALF_SHIFT;
                        exp[0 * HALF_NBINS + bin] = quad_size;
                        exp[1 * HALF_NBINS + bin] = quad_size;
                        exp[2 * HALF_NBINS + bin] = quad_size;
                        return exp;
                    }();
                    constexpr auto *hist_func =
                        traits::template hist_func<true, HALF_BITS, HALF_SHIFT,
                                                   4, 3, 0, 1>;
                    hist_func(quad4_data.data(), quad_mask.data(), size,
                              hist3.data(), 1);
                    CHECK(hist3 == expected3);
                }
            }
        }
    }

    SECTION("2d") {
        SECTION("mono") {
            SECTION("fullbits") {
                std::vector<std::uint32_t> hist(FULL_NBINS);
                auto const expected = [&] {
                    std::vector<std::uint32_t> exp(FULL_NBINS);
                    exp[value_in_roi >> FULL_SHIFT] = quad_size;
                    return exp;
                }();
                SECTION("roi") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, FULL_BITS,
                                                     FULL_SHIFT, 1, 0>;
                    histxy_func(quad_data.data() + quad_y * width + quad_x,
                                nullptr, quad_height, quad_width, width,
                                hist.data(), 1);
                }
                SECTION("mask") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, FULL_BITS,
                                                     FULL_SHIFT, 1, 0>;
                    histxy_func(quad_data.data(), quad_mask.data(), height,
                                width, width, hist.data(), 1);
                }
                SECTION("roi+mask") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, FULL_BITS,
                                                     FULL_SHIFT, 1, 0>;
                    histxy_func(quad_data.data() + quad_y * width + quad_x,
                                quad_mask.data() + quad_y * width + quad_x,
                                quad_height, quad_width, width, hist.data(),
                                1);
                }
                CHECK(hist == expected);
            }
            SECTION("halfbits") {
                std::vector<std::uint32_t> hist(HALF_NBINS);
                auto const expected = [&] {
                    std::vector<std::uint32_t> exp(HALF_NBINS);
                    exp[value_in_roi >> HALF_SHIFT] = quad_size;
                    return exp;
                }();
                SECTION("roi") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, HALF_BITS,
                                                     HALF_SHIFT, 1, 0>;
                    histxy_func(quad_data.data() + quad_y * width + quad_x,
                                nullptr, quad_height, quad_width, width,
                                hist.data(), 1);
                }
                SECTION("mask") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, HALF_BITS,
                                                     HALF_SHIFT, 1, 0>;
                    histxy_func(quad_data.data(), quad_mask.data(), height,
                                width, width, hist.data(), 1);
                }
                SECTION("roi+mask") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, HALF_BITS,
                                                     HALF_SHIFT, 1, 0>;
                    histxy_func(quad_data.data() + quad_y * width + quad_x,
                                quad_mask.data() + quad_y * width + quad_x,
                                quad_height, quad_width, width, hist.data(),
                                1);
                }
                CHECK(hist == expected);
            }
        }
        SECTION("multi") {
            SECTION("fullbits") {
                std::vector<std::uint32_t> hist3(3 * FULL_NBINS);
                auto const expected3 = [&] {
                    std::vector<std::uint32_t> exp(3 * FULL_NBINS);
                    auto const bin = value_in_roi >> FULL_SHIFT;
                    exp[0 * FULL_NBINS + bin] = quad_size;
                    exp[1 * FULL_NBINS + bin] = quad_size;
                    exp[2 * FULL_NBINS + bin] = quad_size;
                    return exp;
                }();
                SECTION("roi") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, FULL_BITS,
                                                     FULL_SHIFT, 4, 3, 0, 1>;
                    histxy_func(quad4_data.data() +
                                    4 * (quad_y * width + quad_x),
                                nullptr, quad_height, quad_width, width,
                                hist3.data(), 1);
                }
                SECTION("mask") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, FULL_BITS,
                                                     FULL_SHIFT, 4, 3, 0, 1>;
                    histxy_func(quad4_data.data(), quad_mask.data(), height,
                                width, width, hist3.data(), 1);
                }
                SECTION("roi+mask") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, FULL_BITS,
                                                     FULL_SHIFT, 4, 3, 0, 1>;
                    histxy_func(
                        quad4_data.data() + 4 * (quad_y * width + quad_x),
                        quad_mask.data() + quad_y * width + quad_x,
                        quad_height, quad_width, width, hist3.data(), 1);
                }
                CHECK(hist3 == expected3);
            }
            SECTION("halfbits") {
                std::vector<std::uint32_t> hist3(3 * HALF_NBINS);
                auto const expected3 = [&] {
                    std::vector<std::uint32_t> exp(3 * HALF_NBINS);
                    auto const bin = value_in_roi >> HALF_SHIFT;
                    exp[0 * HALF_NBINS + bin] = quad_size;
                    exp[1 * HALF_NBINS + bin] = quad_size;
                    exp[2 * HALF_NBINS + bin] = quad_size;
                    return exp;
                }();
                SECTION("roi") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, HALF_BITS,
                                                     HALF_SHIFT, 4, 3, 0, 1>;
                    histxy_func(quad4_data.data() +
                                    4 * (quad_y * width + quad_x),
                                nullptr, quad_height, quad_width, width,
                                hist3.data(), 1);
                }
                SECTION("mask") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, HALF_BITS,
                                                     HALF_SHIFT, 4, 3, 0, 1>;
                    histxy_func(quad4_data.data(), quad_mask.data(), height,
                                width, width, hist3.data(), 1);
                }
                SECTION("roi+mask") {
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, HALF_BITS,
                                                     HALF_SHIFT, 4, 3, 0, 1>;
                    histxy_func(
                        quad4_data.data() + 4 * (quad_y * width + quad_x),
                        quad_mask.data() + quad_y * width + quad_x,
                        quad_height, quad_width, width, hist3.data(), 1);
                }
                CHECK(hist3 == expected3);
            }
        }
    }
}

} // namespace ihist