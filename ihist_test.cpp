/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include <ihist.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ostream>
#include <random>
#include <type_traits>
#include <vector>

namespace ihist {

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

namespace internal {

TEST_CASE("bin_index-full-bits") {
    STATIC_CHECK(bin_index<std::uint8_t>(0) == 0);
    STATIC_CHECK(bin_index<std::uint8_t>(255) == 255);
    STATIC_CHECK(bin_index(std::uint8_t(255)) == 255);
    STATIC_CHECK(bin_index<std::uint16_t>(0) == 0);
    STATIC_CHECK(bin_index<std::uint16_t>(65535) == 65535);
    STATIC_CHECK(bin_index(std::uint16_t(65535)) == 65535);
}

TEST_CASE("bin_index-hi-bits") {
    STATIC_CHECK(bin_index<std::uint16_t, 12>(0x0fff) == 0x0fff);
    STATIC_CHECK(bin_index<std::uint16_t, 12>(0xffff) == 0x1000);
}

TEST_CASE("bin_index-lo-bits") {
    STATIC_CHECK(bin_index<std::uint16_t, 12, 4>(0xfff0) == 0x0fff);
    STATIC_CHECK(bin_index<std::uint16_t, 12, 4>(0xffff) == 0x0fff);
}

TEST_CASE("bin_index-mid-bits") {
    STATIC_CHECK(bin_index<std::uint16_t, 8, 4>(0x0000) == 0);
    STATIC_CHECK(bin_index<std::uint16_t, 8, 4>(0x0010) == 1);
    STATIC_CHECK(bin_index<std::uint16_t, 8, 4>(0x0ff0) == 0xff);
    STATIC_CHECK(bin_index<std::uint16_t, 8, 4>(0x1000) == 256);
    STATIC_CHECK(bin_index<std::uint16_t, 8, 4>(0x1010) == 256);
    STATIC_CHECK(bin_index<std::uint16_t, 8, 4>(0xffff) == 256);
}

TEST_CASE("first_aligned_index_impl") {
    STATIC_CHECK(first_aligned_index_impl<std::uint8_t, 1>(0) == 0);
    STATIC_CHECK(first_aligned_index_impl<std::uint8_t, 1>(1) == 0);
    STATIC_CHECK(first_aligned_index_impl<std::uint8_t, 2>(0) == 0);
    STATIC_CHECK(first_aligned_index_impl<std::uint8_t, 2>(1) == 1);
    STATIC_CHECK(first_aligned_index_impl<std::uint8_t, 2>(2) == 0);
    STATIC_CHECK(first_aligned_index_impl<std::uint8_t, 4>(0) == 0);
    STATIC_CHECK(first_aligned_index_impl<std::uint8_t, 4>(1) == 3);
    STATIC_CHECK(first_aligned_index_impl<std::uint8_t, 4>(2) == 2);
    STATIC_CHECK(first_aligned_index_impl<std::uint8_t, 4>(3) == 1);
    STATIC_CHECK(first_aligned_index_impl<std::uint8_t, 4>(4) == 0);

    STATIC_CHECK(first_aligned_index_impl<std::uint16_t, 1>(0) == 0);
    STATIC_CHECK(first_aligned_index_impl<std::uint16_t, 1>(2) == 0);
    STATIC_CHECK(first_aligned_index_impl<std::uint16_t, 2>(0) == 0);
    STATIC_CHECK(first_aligned_index_impl<std::uint16_t, 2>(2) == 0);
    STATIC_CHECK(first_aligned_index_impl<std::uint16_t, 4>(0) == 0);
    STATIC_CHECK(first_aligned_index_impl<std::uint16_t, 4>(2) == 1);
    STATIC_CHECK(first_aligned_index_impl<std::uint16_t, 4>(4) == 0);
}

} // namespace internal

TEMPLATE_TEST_CASE("empty-data", "", std::uint8_t, std::uint16_t) {
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune1{1};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune3{3};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune01{0, 1};
    static constexpr tuning_parameters tune02{0, 2};
    static constexpr tuning_parameters tune03{0, 3};
    auto const hist_func = GENERATE(
        hist_unoptimized_st<TestType>, hist_striped_st<tune0, TestType>,
        hist_striped_st<tune1, TestType>, hist_striped_st<tune2, TestType>,
        hist_striped_st<tune3, TestType>, hist_striped_st<tune00, TestType>,
        hist_striped_st<tune01, TestType>, hist_striped_st<tune02, TestType>,
        hist_striped_st<tune03, TestType>, hist_unoptimized_mt<TestType>,
        hist_striped_mt<tune0, TestType>, hist_striped_mt<tune1, TestType>,
        hist_striped_mt<tune2, TestType>, hist_striped_mt<tune3, TestType>,
        hist_striped_mt<tune00, TestType>, hist_striped_mt<tune01, TestType>,
        hist_striped_mt<tune02, TestType>, hist_striped_mt<tune03, TestType>);

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(nullptr, nullptr, 0, hist.data(), 1);
    std::array<std::uint32_t, NBINS> ref{};
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("empty-data-xy", "", std::uint8_t, std::uint16_t) {
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune1{1};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune3{3};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune01{0, 1};
    static constexpr tuning_parameters tune02{0, 2};
    static constexpr tuning_parameters tune03{0, 3};
    auto const histxy_func = GENERATE(
        histxy_unoptimized_st<TestType>, histxy_striped_st<tune0, TestType>,
        histxy_striped_st<tune1, TestType>, histxy_striped_st<tune2, TestType>,
        histxy_striped_st<tune3, TestType>,
        histxy_striped_st<tune00, TestType>,
        histxy_striped_st<tune01, TestType>,
        histxy_striped_st<tune02, TestType>,
        histxy_striped_st<tune03, TestType>, histxy_unoptimized_mt<TestType>,
        histxy_striped_mt<tune0, TestType>, histxy_striped_mt<tune1, TestType>,
        histxy_striped_mt<tune2, TestType>, histxy_striped_mt<tune3, TestType>,
        histxy_striped_mt<tune00, TestType>,
        histxy_striped_mt<tune01, TestType>,
        histxy_striped_mt<tune02, TestType>,
        histxy_striped_mt<tune03, TestType>);

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    std::array<std::uint32_t, NBINS> hist{};
    histxy_func(nullptr, nullptr, 0, 0, 0, 0, 0, 0, hist.data(), 1);
    std::array<std::uint32_t, NBINS> ref{};
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("const-data", "", std::uint8_t, std::uint16_t) {
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune1{1};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune3{3};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune01{0, 1};
    static constexpr tuning_parameters tune02{0, 2};
    static constexpr tuning_parameters tune03{0, 3};
    auto const hist_func = GENERATE(
        hist_unoptimized_st<TestType>, hist_striped_st<tune0, TestType>,
        hist_striped_st<tune1, TestType>, hist_striped_st<tune2, TestType>,
        hist_striped_st<tune3, TestType>, hist_striped_st<tune00, TestType>,
        hist_striped_st<tune01, TestType>, hist_striped_st<tune02, TestType>,
        hist_striped_st<tune03, TestType>, hist_unoptimized_mt<TestType>,
        hist_striped_mt<tune0, TestType>, hist_striped_mt<tune1, TestType>,
        hist_striped_mt<tune2, TestType>, hist_striped_mt<tune3, TestType>,
        hist_striped_mt<tune00, TestType>, hist_striped_mt<tune01, TestType>,
        hist_striped_mt<tune02, TestType>, hist_striped_mt<tune03, TestType>);

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    std::size_t size = GENERATE(1, 2, 7, 1000);
    TestType value = GENERATE(0, 1, NBINS - 1);
    CAPTURE(size, value);

    std::vector<TestType> data(size, value);
    std::array<std::uint32_t, NBINS> hist{};
    std::array<std::uint32_t, NBINS> ref{};

    SECTION("malloc-aligned") {
        hist_func(data.data(), nullptr, data.size(), hist.data(), 1);
        ref[value] = size;
        CHECK(hist == ref);
    }

    SECTION("unaligned") {
        hist_func(data.data() + 1, nullptr, data.size() - 1, hist.data(), 1);
        ref[value] = size - 1;
        CHECK(hist == ref);
    }
}

TEMPLATE_TEST_CASE("const-data-xy", "", std::uint8_t, std::uint16_t) {
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune1{1};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune3{3};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune01{0, 1};
    static constexpr tuning_parameters tune02{0, 2};
    static constexpr tuning_parameters tune03{0, 3};
    auto const histxy_func = GENERATE(
        histxy_unoptimized_st<TestType>, histxy_striped_st<tune0, TestType>,
        histxy_striped_st<tune1, TestType>, histxy_striped_st<tune2, TestType>,
        histxy_striped_st<tune3, TestType>,
        histxy_striped_st<tune00, TestType>,
        histxy_striped_st<tune01, TestType>,
        histxy_striped_st<tune02, TestType>,
        histxy_striped_st<tune03, TestType>, histxy_unoptimized_mt<TestType>,
        histxy_striped_mt<tune0, TestType>, histxy_striped_mt<tune1, TestType>,
        histxy_striped_mt<tune2, TestType>, histxy_striped_mt<tune3, TestType>,
        histxy_striped_mt<tune00, TestType>,
        histxy_striped_mt<tune01, TestType>,
        histxy_striped_mt<tune02, TestType>,
        histxy_striped_mt<tune03, TestType>);

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    std::size_t const width = GENERATE(1, 2, 7, 100);
    std::size_t const height = width / 2 + 1;
    TestType value = GENERATE(0, 1, NBINS - 1);
    CAPTURE(width, height, value);

    std::vector<TestType> data(width * height, value);
    std::array<std::uint32_t, NBINS> hist{};
    std::array<std::uint32_t, NBINS> ref{};

    SECTION("full-roi") {
        histxy_func(data.data(), nullptr, width, height, 0, 0, width, height,
                    hist.data(), 1);
        ref[value] = width * height;
        CHECK(hist == ref);
    }

    SECTION("skip-row-0") {
        if (height > 1) {
            histxy_func(data.data(), nullptr, width, height, 0, 1, width,
                        height - 1, hist.data(), 1);
            ref[value] = width * (height - 1);
            CHECK(hist == ref);
        }
    }

    SECTION("skip-row-N") {
        if (height > 1) {
            histxy_func(data.data(), nullptr, width, height, 0, 0, width,
                        height - 1, hist.data(), 1);
            ref[value] = width * (height - 1);
            CHECK(hist == ref);
        }
    }

    SECTION("skip-col-0") {
        if (width > 1) {
            histxy_func(data.data(), nullptr, width, height, 1, 0, width - 1,
                        height, hist.data(), 1);
            ref[value] = (width - 1) * height;
            CHECK(hist == ref);
        }
    }

    SECTION("skip-col-N") {
        if (width > 1) {
            histxy_func(data.data(), nullptr, width, height, 0, 0, width - 1,
                        height, hist.data(), 1);
            ref[value] = (width - 1) * height;
            CHECK(hist == ref);
        }
    }
}

TEMPLATE_TEST_CASE("const-data-hi-bit-filtering-discard-low", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) / 2;
    constexpr auto LO_BIT = BITS / 2;
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune1{1};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune3{3};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune01{0, 1};
    static constexpr tuning_parameters tune02{0, 2};
    static constexpr tuning_parameters tune03{0, 3};
    auto const hist_func =
        GENERATE(hist_unoptimized_st<TestType, false, BITS, LO_BIT, 1, 0>,
                 hist_striped_st<tune0, TestType, false, BITS, LO_BIT, 1, 0>,
                 hist_striped_st<tune1, TestType, false, BITS, LO_BIT, 1, 0>,
                 hist_striped_st<tune2, TestType, false, BITS, LO_BIT, 1, 0>,
                 hist_striped_st<tune3, TestType, false, BITS, LO_BIT, 1, 0>,
                 hist_striped_st<tune00, TestType, false, BITS, LO_BIT, 1, 0>,
                 hist_striped_st<tune01, TestType, false, BITS, LO_BIT, 1, 0>,
                 hist_striped_st<tune02, TestType, false, BITS, LO_BIT, 1, 0>,
                 hist_striped_st<tune03, TestType, false, BITS, LO_BIT, 1, 0>);

    constexpr auto NBINS = 1 << BITS;
    std::size_t size = GENERATE(1, 7, 100);
    TestType lo_bits = GENERATE(0, 1, (1 << LO_BIT) - 1);
    TestType sample = GENERATE(0, 1, NBINS - 1);
    TestType value = (sample << LO_BIT) | lo_bits;
    CAPTURE(size, sample, lo_bits, value);

    SECTION("zero-hi-bits") {
        std::vector<TestType> data(size, value);
        std::array<std::uint32_t, NBINS> hist{};
        hist_func(data.data(), nullptr, data.size(), hist.data(), 0);
        std::array<std::uint32_t, NBINS> ref{};
        ref[sample] = size;
        CHECK(hist == ref);
    }

    SECTION("non-zero-hi-bits") {
        std::vector<TestType> data(size, value | (1 << (BITS + LO_BIT)));
        std::array<std::uint32_t, NBINS> hist{};
        hist_func(data.data(), nullptr, data.size(), hist.data(), 0);
        std::array<std::uint32_t, NBINS> ref{};
        CHECK(hist == ref);
    }
}

TEMPLATE_TEST_CASE("const-data-hi-bit-filtering-discard-low-xy", "",
                   std::uint8_t, std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) / 2;
    constexpr auto LO_BIT = BITS / 2;
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune02{0, 2};
    auto const histxy_func = GENERATE(
        histxy_unoptimized_st<TestType, false, BITS, LO_BIT, 1, 0>,
        histxy_striped_st<tune0, TestType, false, BITS, LO_BIT, 1, 0>,
        histxy_striped_st<tune2, TestType, false, BITS, LO_BIT, 1, 0>,
        histxy_striped_st<tune00, TestType, false, BITS, LO_BIT, 1, 0>,
        histxy_striped_st<tune02, TestType, false, BITS, LO_BIT, 1, 0>);

    constexpr auto NBINS = 1 << BITS;
    std::size_t width = GENERATE(1, 7, 100);
    std::size_t height = width / 2 + 1;
    TestType lo_bits = GENERATE(0, 1, (1 << LO_BIT) - 1);
    TestType sample = GENERATE(0, 1, NBINS - 1);
    TestType value = (sample << LO_BIT) | lo_bits;
    CAPTURE(width, height, sample, lo_bits, value);

    SECTION("zero-hi-bits") {
        std::vector<TestType> data(width * height, value);
        std::array<std::uint32_t, NBINS> hist{};
        histxy_func(data.data(), nullptr, width, height, 0, 0, width, height,
                    hist.data(), 0);
        std::array<std::uint32_t, NBINS> ref{};
        ref[sample] = width * height;
        CHECK(hist == ref);
    }

    SECTION("non-zero-hi-bits") {
        std::vector<TestType> data(width * height,
                                   value | (1 << (BITS + LO_BIT)));
        std::array<std::uint32_t, NBINS> hist{};
        histxy_func(data.data(), nullptr, width, height, 0, 0, width, height,
                    hist.data(), 0);
        std::array<std::uint32_t, NBINS> ref{};
        CHECK(hist == ref);
    }
}

TEMPLATE_TEST_CASE("const-data-multicomponent", "", std::uint8_t,
                   std::uint16_t) {
    constexpr unsigned BITS = 8 * sizeof(TestType);
    constexpr std::size_t STRIDE = 4;
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune1{1};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune3{3};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune01{0, 1};
    static constexpr tuning_parameters tune02{0, 2};
    static constexpr tuning_parameters tune03{0, 3};
    auto const hist_func = GENERATE(
        hist_unoptimized_st<TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune0, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune1, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune2, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune3, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune00, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune01, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune02, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune03, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unoptimized_mt<TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune0, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune1, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune2, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune3, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune00, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune01, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune02, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune03, TestType, false, BITS, 0, STRIDE, 3, 0, 1>);

    constexpr auto NBINS = 1 << BITS;
    std::size_t size = GENERATE(0, 1, 7, 1000);
    TestType value = GENERATE(0, 1, NBINS - 1);

    constexpr std::array components{1, 2, -1, 0}; // Inverse of (3, 0, 1)
    auto const offset = GENERATE(0, 1, 2, 3);
    auto const component = components[offset];

    CAPTURE(size, value, offset, component);

    std::vector<TestType> data(STRIDE * size);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (i % STRIDE == offset) {
            data[i] = value;
        }
    }

    std::vector<std::uint32_t> hist(3 * NBINS);
    hist_func(data.data(), nullptr, size, hist.data(), 1);
    std::vector<std::uint32_t> ref(3 * NBINS);
    for (int c = 0; c < 3; ++c) {
        if (c == component) {
            ref[c * NBINS + value] = size;
        } else {
            ref[c * NBINS + 0] = size;
        }
    }
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("const-data-multicomponent-xy", "", std::uint8_t,
                   std::uint16_t) {
    constexpr unsigned BITS = 8 * sizeof(TestType);
    constexpr std::size_t STRIDE = 4;
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune02{0, 2};
    auto const histxy_func = GENERATE(
        histxy_unoptimized_st<TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_st<tune0, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_st<tune2, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_st<tune00, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_st<tune02, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_unoptimized_mt<TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_mt<tune0, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_mt<tune2, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_mt<tune00, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_mt<tune02, TestType, false, BITS, 0, STRIDE, 3, 0, 1>);

    constexpr auto NBINS = 1 << BITS;
    std::size_t width = GENERATE(0, 1, 7, 1000);
    std::size_t height = width * 2;
    TestType value = GENERATE(0, 1, NBINS - 1);

    constexpr std::array components{1, 2, -1, 0}; // Inverse of (3, 0, 1)
    auto const offset = GENERATE(0, 1, 2, 3);
    auto const component = components[offset];

    CAPTURE(width, height, value, offset, component);

    std::vector<TestType> data(STRIDE * width * height);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (i % STRIDE == offset) {
            data[i] = value;
        }
    }

    std::vector<std::uint32_t> hist(3 * NBINS);
    histxy_func(data.data(), nullptr, width, height, 0, 0, width, height,
                hist.data(), 1);
    std::vector<std::uint32_t> ref(3 * NBINS);
    for (int c = 0; c < 3; ++c) {
        if (c == component) {
            ref[c * NBINS + value] = width * height;
        } else {
            ref[c * NBINS + 0] = width * height;
        }
    }
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data", "", std::uint8_t, std::uint16_t) {
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune1{1};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune3{3};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune01{0, 1};
    static constexpr tuning_parameters tune02{0, 2};
    static constexpr tuning_parameters tune03{0, 3};
    auto const hist_func = GENERATE(
        hist_unoptimized_st<TestType>, hist_striped_st<tune0, TestType>,
        hist_striped_st<tune1, TestType>, hist_striped_st<tune2, TestType>,
        hist_striped_st<tune3, TestType>, hist_striped_st<tune00, TestType>,
        hist_striped_st<tune01, TestType>, hist_striped_st<tune02, TestType>,
        hist_striped_st<tune03, TestType>, hist_unoptimized_mt<TestType>,
        hist_striped_mt<tune0, TestType>, hist_striped_mt<tune1, TestType>,
        hist_striped_mt<tune2, TestType>, hist_striped_mt<tune3, TestType>,
        hist_striped_mt<tune00, TestType>, hist_striped_mt<tune01, TestType>,
        hist_striped_mt<tune02, TestType>, hist_striped_mt<tune03, TestType>);
    auto const ref_func = hist_unoptimized_st<TestType>;

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    auto const data = test_data<TestType>(1 << (20 - sizeof(TestType)));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), nullptr, data.size(), hist.data(), 1);
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(data.data(), nullptr, data.size(), ref.data(), 1);
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-xy", "", std::uint8_t, std::uint16_t) {
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune02{0, 2};
    auto const histxy_func = GENERATE(
        histxy_unoptimized_st<TestType>, histxy_striped_st<tune0, TestType>,
        histxy_striped_st<tune2, TestType>,
        histxy_striped_st<tune00, TestType>,
        histxy_striped_st<tune02, TestType>, histxy_unoptimized_mt<TestType>,
        histxy_striped_mt<tune0, TestType>, histxy_striped_mt<tune2, TestType>,
        histxy_striped_mt<tune00, TestType>,
        histxy_striped_mt<tune02, TestType>);
    auto const ref_func = hist_unoptimized_st<TestType>;

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    constexpr std::size_t width = 1 << 9;
    constexpr std::size_t height = 1 << 8;
    auto const data = test_data<TestType>(width * height);
    std::array<std::uint32_t, NBINS> hist{};
    histxy_func(data.data(), nullptr, width, height, 0, 0, width, height,
                hist.data(), 1);
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(data.data(), nullptr, data.size(), ref.data(), 1);
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-clean-filtered", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) - 4;
    auto hist_func =
        GENERATE(hist_unoptimized_st<TestType, false, BITS>,
                 hist_striped_st<untuned_parameters, TestType, false, BITS>,
                 hist_unoptimized_mt<TestType, false, BITS>,
                 hist_striped_mt<untuned_parameters, TestType, false, BITS>);
    auto ref_func = hist_unoptimized_st<TestType, false, BITS>;

    constexpr auto NBINS = 1 << BITS;
    // Test with 4/12-bit data with no spurious high bits:
    auto const data = test_data<TestType, BITS>(1 << (20 - sizeof(TestType)));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), nullptr, data.size(), hist.data(), 1);
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(data.data(), nullptr, data.size(), ref.data(), 1);
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-clean-filtered-xy", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) - 4;
    auto histxy_func =
        GENERATE(histxy_unoptimized_st<TestType, false, BITS>,
                 histxy_striped_st<untuned_parameters, TestType, false, BITS>,
                 histxy_unoptimized_mt<TestType, false, BITS>,
                 histxy_striped_mt<untuned_parameters, TestType, false, BITS>);
    auto ref_func = hist_unoptimized_st<TestType, false, BITS>;

    constexpr auto NBINS = 1 << BITS;
    // Test with 4/12-bit data with no spurious high bits:
    constexpr std::size_t width = 1 << 9;
    constexpr std::size_t height = 1 << 8;
    auto const data = test_data<TestType, BITS>(width * height);
    std::array<std::uint32_t, NBINS> hist{};
    histxy_func(data.data(), nullptr, width, height, 0, 0, width, height,
                hist.data(), 1);
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(data.data(), nullptr, data.size(), ref.data(), 1);
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-clean-discard-low", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) / 2;
    constexpr auto LO_BIT = BITS / 2;
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune1{1};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune3{3};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune01{0, 1};
    static constexpr tuning_parameters tune02{0, 2};
    static constexpr tuning_parameters tune03{0, 3};
    auto const hist_func =
        GENERATE(hist_unoptimized_st<TestType, false, BITS, LO_BIT>,
                 hist_striped_st<tune0, TestType, false, BITS, LO_BIT>,
                 hist_striped_st<tune1, TestType, false, BITS, LO_BIT>,
                 hist_striped_st<tune2, TestType, false, BITS, LO_BIT>,
                 hist_striped_st<tune3, TestType, false, BITS, LO_BIT>,
                 hist_striped_st<tune00, TestType, false, BITS, LO_BIT>,
                 hist_striped_st<tune01, TestType, false, BITS, LO_BIT>,
                 hist_striped_st<tune02, TestType, false, BITS, LO_BIT>,
                 hist_striped_st<tune03, TestType, false, BITS, LO_BIT>,
                 hist_unoptimized_mt<TestType, false, BITS, LO_BIT>,
                 hist_striped_mt<tune0, TestType, false, BITS, LO_BIT>,
                 hist_striped_mt<tune1, TestType, false, BITS, LO_BIT>,
                 hist_striped_mt<tune2, TestType, false, BITS, LO_BIT>,
                 hist_striped_mt<tune3, TestType, false, BITS, LO_BIT>,
                 hist_striped_mt<tune00, TestType, false, BITS, LO_BIT>,
                 hist_striped_mt<tune01, TestType, false, BITS, LO_BIT>,
                 hist_striped_mt<tune02, TestType, false, BITS, LO_BIT>,
                 hist_striped_mt<tune03, TestType, false, BITS, LO_BIT>);
    auto ref_func = hist_unoptimized_st<TestType, false, BITS, LO_BIT>;

    constexpr auto NBINS = 1 << BITS;
    // Test with 6/12-bit data with no spurious high bits (but random low
    // bits):
    auto const data =
        test_data<TestType, BITS + LO_BIT>(1 << (20 - sizeof(TestType)));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), nullptr, data.size(), hist.data(), 1);
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(data.data(), nullptr, data.size(), ref.data(), 1);
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-clean-discard-low-xy", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) / 2;
    constexpr auto LO_BIT = BITS / 2;
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune02{0, 2};
    auto const histxy_func =
        GENERATE(histxy_unoptimized_st<TestType, false, BITS, LO_BIT>,
                 histxy_striped_st<tune0, TestType, false, BITS, LO_BIT>,
                 histxy_striped_st<tune2, TestType, false, BITS, LO_BIT>,
                 histxy_striped_st<tune00, TestType, false, BITS, LO_BIT>,
                 histxy_striped_st<tune02, TestType, false, BITS, LO_BIT>,
                 histxy_unoptimized_mt<TestType, false, BITS, LO_BIT>,
                 histxy_striped_mt<tune0, TestType, false, BITS, LO_BIT>,
                 histxy_striped_mt<tune2, TestType, false, BITS, LO_BIT>,
                 histxy_striped_mt<tune00, TestType, false, BITS, LO_BIT>,
                 histxy_striped_mt<tune02, TestType, false, BITS, LO_BIT>);
    auto ref_func = hist_unoptimized_st<TestType, false, BITS, LO_BIT>;

    constexpr auto NBINS = 1 << BITS;
    // Test with 6/12-bit data with no spurious high bits (but random low
    // bits):
    constexpr std::size_t width = 1 << 9;
    constexpr std::size_t height = 1 << 8;
    auto const data = test_data<TestType, BITS + LO_BIT>(width * height);
    std::array<std::uint32_t, NBINS> hist{};
    histxy_func(data.data(), nullptr, width, height, 0, 0, width, height,
                hist.data(), 1);
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(data.data(), nullptr, data.size(), ref.data(), 1);
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-unclean-filtered", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) - 4;
    auto hist_func =
        GENERATE(hist_unoptimized_st<TestType, false, BITS>,
                 hist_striped_st<untuned_parameters, TestType, false, BITS>,
                 hist_unoptimized_mt<TestType, false, BITS>,
                 hist_striped_mt<untuned_parameters, TestType, false, BITS>);
    auto ref_func = hist_unoptimized_st<TestType, false, BITS>;

    constexpr auto NBINS = 1 << BITS;

    // Test with 6/14-bit data, exceeding the histogram range:
    auto const data =
        test_data<TestType, BITS + 2>(1 << (20 - sizeof(TestType)));

    // Cleaned data (limited to values in 4/12-bit range) should produce same
    // result as hi_to_match=0.
    auto const clean_data = [&] {
        std::vector<TestType> clean;
        clean.reserve(data.size() / 3);
        std::copy_if(data.begin(), data.end(), std::back_inserter(clean),
                     [&](TestType v) { return v < (1 << BITS); });
        return clean;
    }();

    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), nullptr, data.size(), hist.data(), 1);
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(clean_data.data(), nullptr, clean_data.size(), ref.data(), 1);
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-unclean-filtered-xy", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) - 4;
    auto histxy_func =
        GENERATE(histxy_unoptimized_st<TestType, false, BITS>,
                 histxy_striped_st<untuned_parameters, TestType, false, BITS>,
                 histxy_unoptimized_mt<TestType, false, BITS>,
                 histxy_striped_mt<untuned_parameters, TestType, false, BITS>);
    auto ref_func = hist_unoptimized_st<TestType, false, BITS>;

    constexpr auto NBINS = 1 << BITS;

    // Test with 6/14-bit data, exceeding the histogram range:
    constexpr std::size_t width = 1 << 9;
    constexpr std::size_t height = 1 << 8;
    auto const data = test_data<TestType, BITS + 2>(width * height);

    // Cleaned data (limited to values in 4/12-bit range) should produce same
    // result as hi_to_match=0.
    auto const clean_data = [&] {
        std::vector<TestType> clean;
        clean.reserve(data.size() / 3);
        std::copy_if(data.begin(), data.end(), std::back_inserter(clean),
                     [&](TestType v) { return v < (1 << BITS); });
        return clean;
    }();

    std::array<std::uint32_t, NBINS> hist{};
    histxy_func(data.data(), nullptr, width, height, 0, 0, width, height,
                hist.data(), 1);
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(clean_data.data(), nullptr, clean_data.size(), ref.data(), 1);
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-multicomponent", "", std::uint8_t,
                   std::uint16_t) {
    constexpr unsigned BITS = 8 * sizeof(TestType);
    constexpr std::size_t STRIDE = 4;
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune1{1};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune3{3};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune01{0, 1};
    static constexpr tuning_parameters tune02{0, 2};
    static constexpr tuning_parameters tune03{0, 3};
    auto const hist_func = GENERATE(
        hist_unoptimized_st<TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune0, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune1, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune2, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune3, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune00, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune01, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune02, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_st<tune03, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unoptimized_mt<TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune0, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune1, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune2, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune3, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune00, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune01, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune02, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        hist_striped_mt<tune03, TestType, false, BITS, 0, STRIDE, 3, 0, 1>);
    auto const ref_func =
        hist_unoptimized_st<TestType, false, BITS, 0, STRIDE, 3, 0, 1>;

    constexpr auto NBINS = 1 << BITS;
    auto const data = test_data<TestType>(STRIDE << (20 - sizeof(TestType)));
    std::vector<std::uint32_t> hist(3 * NBINS);
    hist_func(data.data(), nullptr, data.size() / STRIDE, hist.data(), 1);
    std::vector<std::uint32_t> ref(3 * NBINS);
    ref_func(data.data(), nullptr, data.size() / STRIDE, ref.data(), 1);
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-multicomponent-xy", "", std::uint8_t,
                   std::uint16_t) {
    constexpr unsigned BITS = 8 * sizeof(TestType);
    constexpr std::size_t STRIDE = 4;
    static constexpr tuning_parameters tune0{0};
    static constexpr tuning_parameters tune2{2};
    static constexpr tuning_parameters tune00{0, 0};
    static constexpr tuning_parameters tune02{0, 2};
    auto const histxy_func = GENERATE(
        histxy_unoptimized_st<TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_st<tune0, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_st<tune2, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_st<tune00, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_st<tune02, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_unoptimized_mt<TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_mt<tune0, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_mt<tune2, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_mt<tune00, TestType, false, BITS, 0, STRIDE, 3, 0, 1>,
        histxy_striped_mt<tune02, TestType, false, BITS, 0, STRIDE, 3, 0, 1>);
    auto const ref_func =
        hist_unoptimized_st<TestType, false, BITS, 0, STRIDE, 3, 0, 1>;

    constexpr auto NBINS = 1 << BITS;
    constexpr std::size_t width = 1 << 9;
    constexpr std::size_t height = 1 << 8;
    auto const data = test_data<TestType>(STRIDE * width * height);
    std::vector<std::uint32_t> hist(3 * NBINS);
    histxy_func(data.data(), nullptr, width, height, 0, 0, width, height,
                hist.data(), 1);
    std::vector<std::uint32_t> ref(3 * NBINS);
    ref_func(data.data(), nullptr, width * height, ref.data(), 1);
    CHECK(hist == ref);
}

} // namespace ihist