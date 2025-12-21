/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include <ihist.hpp>

#include "parameterization.hpp"
#include "test_data.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

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

    auto const mask = test_data<std::uint8_t, 1>(size);

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
                    refxy_func(data.data() + quad_y * width + quad_x, nullptr,
                               quad_height, quad_width, width, width,
                               ref.data(), 1);

                    std::vector<std::uint32_t> hist(FULL_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, FULL_BITS,
                                                     FULL_SHIFT, 1, 0>;
                    histxy_func(data.data() + quad_y * width + quad_x, nullptr,
                                quad_height, quad_width, width, width,
                                hist.data(), 1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(FULL_NBINS);
                    constexpr auto *refxy_func =
                        histxy_unoptimized_st<T, true>;
                    refxy_func(data.data() + quad_y * width + quad_x,
                               mask.data() + quad_y * width + quad_x,
                               quad_height, quad_width, width, width,
                               ref.data(), 1);

                    std::vector<std::uint32_t> hist(FULL_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, FULL_BITS,
                                                     FULL_SHIFT, 1, 0>;
                    histxy_func(data.data() + quad_y * width + quad_x,
                                mask.data() + quad_y * width + quad_x,
                                quad_height, quad_width, width, width,
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
                    refxy_func(data.data() + quad_y * width + quad_x, nullptr,
                               quad_height, quad_width, width, width,
                               ref.data(), 1);

                    std::vector<std::uint32_t> hist(HALF_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, HALF_BITS,
                                                     HALF_SHIFT, 1, 0>;
                    histxy_func(data.data() + quad_y * width + quad_x, nullptr,
                                quad_height, quad_width, width, width,
                                hist.data(), 1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(HALF_NBINS);
                    constexpr auto *refxy_func =
                        histxy_unoptimized_st<T, true, HALF_BITS, HALF_SHIFT>;
                    refxy_func(data.data() + quad_y * width + quad_x,
                               mask.data() + quad_y * width + quad_x,
                               quad_height, quad_width, width, width,
                               ref.data(), 1);

                    std::vector<std::uint32_t> hist(HALF_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, HALF_BITS,
                                                     HALF_SHIFT, 1, 0>;
                    histxy_func(data.data() + quad_y * width + quad_x,
                                mask.data() + quad_y * width + quad_x,
                                quad_height, quad_width, width, width,
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
                    refxy_func(data.data() + 4 * (quad_y * width + quad_x),
                               nullptr, quad_height, quad_width, width, width,
                               ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * FULL_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, FULL_BITS,
                                                     FULL_SHIFT, 4, 3, 0, 1>;
                    histxy_func(data.data() + 4 * (quad_y * width + quad_x),
                                nullptr, quad_height, quad_width, width, width,
                                hist.data(), 1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(3 * FULL_NBINS);
                    constexpr auto *refxy_func =
                        histxy_unoptimized_st<T, true, FULL_BITS, FULL_SHIFT,
                                              4, 3, 0, 1>;
                    refxy_func(data.data() + 4 * (quad_y * width + quad_x),
                               mask.data() + quad_y * width + quad_x,
                               quad_height, quad_width, width, width,
                               ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * FULL_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, FULL_BITS,
                                                     FULL_SHIFT, 4, 3, 0, 1>;
                    histxy_func(data.data() + 4 * (quad_y * width + quad_x),
                                mask.data() + quad_y * width + quad_x,
                                quad_height, quad_width, width, width,
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
                    refxy_func(data.data() + 4 * (quad_y * width + quad_x),
                               nullptr, quad_height, quad_width, width, width,
                               ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * HALF_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<false, HALF_BITS,
                                                     HALF_SHIFT, 4, 3, 0, 1>;
                    histxy_func(data.data() + 4 * (quad_y * width + quad_x),
                                nullptr, quad_height, quad_width, width, width,
                                hist.data(), 1);
                    CHECK(hist == ref);
                }
                SECTION("mask") {
                    std::vector<std::uint32_t> ref(3 * HALF_NBINS);
                    constexpr auto *refxy_func =
                        histxy_unoptimized_st<T, true, HALF_BITS, HALF_SHIFT,
                                              4, 3, 0, 1>;
                    refxy_func(data.data() + 4 * (quad_y * width + quad_x),
                               mask.data() + quad_y * width + quad_x,
                               quad_height, quad_width, width, width,
                               ref.data(), 1);

                    std::vector<std::uint32_t> hist(3 * HALF_NBINS);
                    constexpr auto *histxy_func =
                        traits::template histxy_func<true, HALF_BITS,
                                                     HALF_SHIFT, 4, 3, 0, 1>;
                    histxy_func(data.data() + 4 * (quad_y * width + quad_x),
                                mask.data() + quad_y * width + quad_x,
                                quad_height, quad_width, width, width,
                                hist.data(), 1);
                    CHECK(hist == ref);
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE("dynamic-random-input", "", dynamic_test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    constexpr std::size_t indices[] = {0, 1};
    constexpr std::size_t width = 65;
    constexpr std::size_t height = 63;
    constexpr std::size_t quad_x = 7;
    constexpr std::size_t quad_y = 5;
    constexpr std::size_t quad_width = 33;
    constexpr std::size_t quad_height = 29;
    constexpr std::size_t size = width * height;

    constexpr auto FULL_BITS = 8 * sizeof(T);
    constexpr auto FULL_NBINS = 1 << FULL_BITS;
    constexpr auto HALF_BITS = FULL_BITS / 2;
    constexpr auto HALF_NBINS = 1 << HALF_BITS;
    constexpr auto HALF_SHIFT = FULL_BITS / 4;

    auto const mask = test_data<std::uint8_t, 1>(size);
    auto const data = test_data<T>(2 * size);

    SECTION("fullbits") {
        SECTION("nomask") {
            std::vector<std::uint32_t> ref(2 * FULL_NBINS);
            constexpr auto *refxy_func =
                histxy_unoptimized_st<T, false, FULL_BITS, 0, 2, 0, 1>;
            refxy_func(data.data() + 2 * (quad_y * width + quad_x), nullptr,
                       quad_height, quad_width, width, width, ref.data(), 1);

            std::vector<std::uint32_t> hist(2 * FULL_NBINS);
            traits::template histxy_dynamic<false, FULL_BITS, 0>(
                data.data() + 2 * (quad_y * width + quad_x), nullptr,
                quad_height, quad_width, width, width, 2, 2, indices,
                hist.data());
            CHECK(hist == ref);
        }
        SECTION("mask") {
            std::vector<std::uint32_t> ref(2 * FULL_NBINS);
            constexpr auto *refxy_func =
                histxy_unoptimized_st<T, true, FULL_BITS, 0, 2, 0, 1>;
            refxy_func(data.data() + 2 * (quad_y * width + quad_x),
                       mask.data() + quad_y * width + quad_x, quad_height,
                       quad_width, width, width, ref.data(), 1);

            std::vector<std::uint32_t> hist(2 * FULL_NBINS);
            traits::template histxy_dynamic<true, FULL_BITS, 0>(
                data.data() + 2 * (quad_y * width + quad_x),
                mask.data() + quad_y * width + quad_x, quad_height, quad_width,
                width, width, 2, 2, indices, hist.data());
            CHECK(hist == ref);
        }
    }

    SECTION("halfbits") {
        SECTION("nomask") {
            std::vector<std::uint32_t> ref(2 * HALF_NBINS);
            constexpr auto *refxy_func =
                histxy_unoptimized_st<T, false, HALF_BITS, HALF_SHIFT, 2, 0,
                                      1>;
            refxy_func(data.data() + 2 * (quad_y * width + quad_x), nullptr,
                       quad_height, quad_width, width, width, ref.data(), 1);

            std::vector<std::uint32_t> hist(2 * HALF_NBINS);
            traits::template histxy_dynamic<false, HALF_BITS, HALF_SHIFT>(
                data.data() + 2 * (quad_y * width + quad_x), nullptr,
                quad_height, quad_width, width, width, 2, 2, indices,
                hist.data());
            CHECK(hist == ref);
        }
        SECTION("mask") {
            std::vector<std::uint32_t> ref(2 * HALF_NBINS);
            constexpr auto *refxy_func =
                histxy_unoptimized_st<T, true, HALF_BITS, HALF_SHIFT, 2, 0, 1>;
            refxy_func(data.data() + 2 * (quad_y * width + quad_x),
                       mask.data() + quad_y * width + quad_x, quad_height,
                       quad_width, width, width, ref.data(), 1);

            std::vector<std::uint32_t> hist(2 * HALF_NBINS);
            traits::template histxy_dynamic<true, HALF_BITS, HALF_SHIFT>(
                data.data() + 2 * (quad_y * width + quad_x),
                mask.data() + quad_y * width + quad_x, quad_height, quad_width,
                width, width, 2, 2, indices, hist.data());
            CHECK(hist == ref);
        }
    }
}

} // namespace ihist