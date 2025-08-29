/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include <ihist.hpp>

#include "parameterization.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

namespace {

// This produces portably deterministic data given the same seed.
template <typename T, unsigned Bits = 8 * sizeof(T)>
auto generate_random_data(std::size_t count, std::uint32_t seed)
    -> std::vector<T> {
    static_assert(std::is_integral_v<T>);
    static_assert(sizeof(T) <= 8);
    static_assert(Bits <= 8 * sizeof(T));

    // We cannot use std::uniform_int_distribution because it may behave
    // differently depending on the platform, and also does not support 8-bit
    // integers. Instead, we take the low bits.

    std::mt19937_64 engine(seed);
    std::vector<T> data;
    data.resize(count);
    constexpr auto MASK = (1uLL << Bits) - 1;
    std::generate(data.begin(), data.end(),
                  [&] { return static_cast<T>(engine() & MASK); });
    return data;
}

// Reproducible tests!
constexpr std::uint32_t TEST_SEED = 1343208745u;

template <typename T, unsigned Bits = 8 * sizeof(T)>
auto test_data(std::size_t count) -> std::vector<T> {
    return generate_random_data<T, Bits>(count, TEST_SEED);
}

} // namespace

namespace ihist {

// This test trusts our reference (unoptimized) implementation (tested well in
// constant-input and all-values tests) and makes sure the optimized
// implementations produce the same result on large-ish input data.

TEMPLATE_LIST_TEST_CASE("random-input", "", test_traits_list) {
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

    constexpr std::size_t width = 65;
    constexpr std::size_t height = 63;
    constexpr std::size_t quad_x = 7;
    constexpr std::size_t quad_y = 5;
    constexpr std::size_t quad_width = 33;
    constexpr std::size_t quad_height = 29;
    constexpr std::size_t size = width * height;

    auto const mask = test_data<std::uint8_t>(size);

    SECTION("mono") {
        auto const data = test_data<T>(size);
        SECTION("fullbits") {
            SECTION("1d") {
                SECTION("nomask") {
                    std::vector<std::uint32_t> ref(FULL_NBINS);
                    constexpr auto *ref_func = hist_unoptimized_st<T>;
                    ref_func(data.data(), nullptr, size, ref.data(), 1);

                    std::vector<std::uint32_t> hist(FULL_NBINS);
                    constexpr auto *hist_func =
                        traits::template hist_func<false, FULL_BITS,
                                                   FULL_SHIFT, 1, 0>;
                    hist_func(data.data(), nullptr, size, hist.data(), 1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(FULL_NBINS);
                    constexpr auto *ref_func = hist_unoptimized_st<T, true>;
                    ref_func(data.data(), mask.data(), size, ref.data(), 1);

                    std::vector<std::uint32_t> hist(FULL_NBINS);
                    constexpr auto *hist_func =
                        traits::template hist_func<true, FULL_BITS, FULL_SHIFT,
                                                   1, 0>;
                    hist_func(data.data(), mask.data(), size, hist.data(), 1);
                    CHECK(hist == ref);
                }
            }
            SECTION("2d") {
                SECTION("nomask") {
                    std::vector<std::uint32_t> ref(FULL_NBINS);
                    constexpr auto *refxy_func = histxy_unoptimized_st<T>;
                    refxy_func(data.data(), nullptr, width, height, quad_x,
                               quad_y, quad_width, quad_height, ref.data(), 1);

                    std::vector<std::uint32_t> hist(FULL_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, FULL_BITS,
                                                     FULL_SHIFT, 1, 0>;
                    histxy_func(data.data(), nullptr, width, height, quad_x,
                                quad_y, quad_width, quad_height, hist.data(),
                                1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(FULL_NBINS);
                    constexpr auto *refxy_func =
                        histxy_unoptimized_st<T, true>;
                    refxy_func(data.data(), mask.data(), width, height, quad_x,
                               quad_y, quad_width, quad_height, ref.data(), 1);

                    std::vector<std::uint32_t> hist(FULL_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, FULL_BITS,
                                                     FULL_SHIFT, 1, 0>;
                    histxy_func(data.data(), mask.data(), width, height,
                                quad_x, quad_y, quad_width, quad_height,
                                hist.data(), 1);
                    CHECK(hist == ref);
                }
            }
        }
        SECTION("halfbits") {
            SECTION("1d") {
                SECTION("nomask") {
                    std::vector<std::uint32_t> ref(HALF_NBINS);
                    constexpr auto *ref_func =
                        hist_unoptimized_st<T, false, HALF_BITS, HALF_SHIFT>;
                    ref_func(data.data(), nullptr, size, ref.data(), 1);

                    std::vector<std::uint32_t> hist(HALF_NBINS);
                    constexpr auto *hist_func =
                        traits::template hist_func<false, HALF_BITS,
                                                   HALF_SHIFT, 1, 0>;
                    hist_func(data.data(), nullptr, size, hist.data(), 1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(HALF_NBINS);
                    constexpr auto *ref_func =
                        hist_unoptimized_st<T, true, HALF_BITS, HALF_SHIFT>;
                    ref_func(data.data(), mask.data(), size, ref.data(), 1);

                    std::vector<std::uint32_t> hist(HALF_NBINS);
                    constexpr auto *hist_func =
                        traits::template hist_func<true, HALF_BITS, HALF_SHIFT,
                                                   1, 0>;
                    hist_func(data.data(), mask.data(), size, hist.data(), 1);
                    CHECK(hist == ref);
                }
            }
            SECTION("2d") {
                SECTION("nomask") {
                    std::vector<std::uint32_t> ref(HALF_NBINS);
                    constexpr auto *refxy_func =
                        histxy_unoptimized_st<T, false, HALF_BITS, HALF_SHIFT>;
                    refxy_func(data.data(), nullptr, width, height, quad_x,
                               quad_y, quad_width, quad_height, ref.data(), 1);

                    std::vector<std::uint32_t> hist(HALF_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, HALF_BITS,
                                                     HALF_SHIFT, 1, 0>;
                    histxy_func(data.data(), nullptr, width, height, quad_x,
                                quad_y, quad_width, quad_height, hist.data(),
                                1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(HALF_NBINS);
                    constexpr auto *refxy_func =
                        histxy_unoptimized_st<T, true, HALF_BITS, HALF_SHIFT>;
                    refxy_func(data.data(), mask.data(), width, height, quad_x,
                               quad_y, quad_width, quad_height, ref.data(), 1);

                    std::vector<std::uint32_t> hist(HALF_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, HALF_BITS,
                                                     HALF_SHIFT, 1, 0>;
                    histxy_func(data.data(), mask.data(), width, height,
                                quad_x, quad_y, quad_width, quad_height,
                                hist.data(), 1);
                    CHECK(hist == ref);
                }
            }
        }
    }

    SECTION("multi") {
        auto const data = test_data<T>(4 * size);
        SECTION("fullbits") {
            SECTION("1d") {
                SECTION("nomask") {
                    std::vector<std::uint32_t> ref(3 * FULL_NBINS);
                    constexpr auto *ref_func =
                        hist_unoptimized_st<T, false, FULL_BITS, FULL_SHIFT, 4,
                                            3, 0, 1>;
                    ref_func(data.data(), nullptr, size, ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * FULL_NBINS);
                    constexpr auto *hist_func =
                        traits::template hist_func<false, FULL_BITS,
                                                   FULL_SHIFT, 4, 3, 0, 1>;
                    hist_func(data.data(), nullptr, size, hist.data(), 1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(3 * FULL_NBINS);
                    constexpr auto *ref_func =
                        hist_unoptimized_st<T, true, FULL_BITS, FULL_SHIFT, 4,
                                            3, 0, 1>;
                    ref_func(data.data(), mask.data(), size, ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * FULL_NBINS);
                    constexpr auto *hist_func =
                        traits::template hist_func<true, FULL_BITS, FULL_SHIFT,
                                                   4, 3, 0, 1>;
                    hist_func(data.data(), mask.data(), size, hist.data(), 1);
                    CHECK(hist == ref);
                }
            }
            SECTION("2d") {
                SECTION("nomask") {
                    std::vector<std::uint32_t> ref(3 * FULL_NBINS);
                    constexpr auto *refxy_func =
                        histxy_unoptimized_st<T, false, FULL_BITS, FULL_SHIFT,
                                              4, 3, 0, 1>;
                    refxy_func(data.data(), nullptr, width, height, quad_x,
                               quad_y, quad_width, quad_height, ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * FULL_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, FULL_BITS,
                                                     FULL_SHIFT, 4, 3, 0, 1>;
                    histxy_func(data.data(), nullptr, width, height, quad_x,
                                quad_y, quad_width, quad_height, hist.data(),
                                1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(3 * FULL_NBINS);
                    constexpr auto *refxy_func =
                        histxy_unoptimized_st<T, true, FULL_BITS, FULL_SHIFT,
                                              4, 3, 0, 1>;
                    refxy_func(data.data(), mask.data(), width, height, quad_x,
                               quad_y, quad_width, quad_height, ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * FULL_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, FULL_BITS,
                                                     FULL_SHIFT, 4, 3, 0, 1>;
                    histxy_func(data.data(), mask.data(), width, height,
                                quad_x, quad_y, quad_width, quad_height,
                                hist.data(), 1);
                    CHECK(hist == ref);
                }
            }
        }
        SECTION("halfbits") {
            SECTION("1d") {
                SECTION("nomask") {
                    std::vector<std::uint32_t> ref(3 * HALF_NBINS);
                    constexpr auto *ref_func =
                        hist_unoptimized_st<T, false, HALF_BITS, HALF_SHIFT, 4,
                                            3, 0, 1>;
                    ref_func(data.data(), nullptr, size, ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * HALF_NBINS);
                    constexpr auto *hist_func =
                        traits::template hist_func<false, HALF_BITS,
                                                   HALF_SHIFT, 4, 3, 0, 1>;
                    hist_func(data.data(), nullptr, size, hist.data(), 1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(3 * HALF_NBINS);
                    constexpr auto *ref_func =
                        hist_unoptimized_st<T, true, HALF_BITS, HALF_SHIFT, 4,
                                            3, 0, 1>;
                    ref_func(data.data(), mask.data(), size, ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * HALF_NBINS);
                    constexpr auto *hist_func =
                        traits::template hist_func<true, HALF_BITS, HALF_SHIFT,
                                                   4, 3, 0, 1>;
                    hist_func(data.data(), mask.data(), size, hist.data(), 1);
                    CHECK(hist == ref);
                }
            }
            SECTION("2d") {
                SECTION("nomask") {
                    std::vector<std::uint32_t> ref(3 * HALF_NBINS);
                    constexpr auto *refxy_func =
                        histxy_unoptimized_st<T, false, HALF_BITS, HALF_SHIFT,
                                              4, 3, 0, 1>;
                    refxy_func(data.data(), nullptr, width, height, quad_x,
                               quad_y, quad_width, quad_height, ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * HALF_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, HALF_BITS,
                                                     HALF_SHIFT, 4, 3, 0, 1>;
                    histxy_func(data.data(), nullptr, width, height, quad_x,
                                quad_y, quad_width, quad_height, hist.data(),
                                1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(3 * HALF_NBINS);
                    constexpr auto *refxy_func =
                        histxy_unoptimized_st<T, true, HALF_BITS, HALF_SHIFT,
                                              4, 3, 0, 1>;
                    refxy_func(data.data(), mask.data(), width, height, quad_x,
                               quad_y, quad_width, quad_height, ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * HALF_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, HALF_BITS,
                                                     HALF_SHIFT, 4, 3, 0, 1>;
                    histxy_func(data.data(), mask.data(), width, height,
                                quad_x, quad_y, quad_width, quad_height,
                                hist.data(), 1);
                    CHECK(hist == ref);
                }
            }
        }
    }
}

} // namespace ihist