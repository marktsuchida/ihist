/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include <ihist/ihist.h>

#include "ihist.hpp"

#include "test_data.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <cstdint>

namespace {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

constexpr std::size_t width = 65;
constexpr std::size_t height = 63;
constexpr std::size_t roi_x = 7;
constexpr std::size_t roi_y = 5;
constexpr std::size_t roi_width = 33;
constexpr std::size_t roi_height = 29;
constexpr std::size_t size = width * height;

} // namespace

// We mainly want to test that the API functions dispatch to the correct
// implementations, and the implementations themselves are tested elsewhere.
// So use random input and check for match with unoptimized implementation.

#define TEST_CASE_MONO(format_bits, sample_bits)                              \
    TEST_CASE("mono" #format_bits "-" #sample_bits) {                         \
        auto const data = test_data<u##format_bits, sample_bits>(size);       \
        auto const mask = test_data<u8, 1>(size);                             \
        constexpr std::size_t NBINS = 1 << sample_bits;                       \
        std::vector<u32> hist(NBINS);                                         \
        std::vector<u32> ref(NBINS);                                          \
        constexpr std::size_t indices[] = {0};                                \
                                                                              \
        bool const parallel = GENERATE(false, true);                          \
                                                                              \
        SECTION("nomask") {                                                   \
            ihist::histxy_unoptimized_st<u##format_bits, false, sample_bits,  \
                                         0, 1, 0>(                            \
                data.data() + roi_y * width + roi_x, nullptr, roi_height,     \
                roi_width, width, ref.data());                                \
            ihist_hist##format_bits##_2d(                                     \
                sample_bits, data.data() + roi_y * width + roi_x, nullptr,    \
                roi_height, roi_width, width, 1, 1, indices, hist.data(),     \
                parallel);                                                    \
        }                                                                     \
        SECTION("mask") {                                                     \
            ihist::histxy_unoptimized_st<u##format_bits, true, sample_bits,   \
                                         0, 1, 0>(                            \
                data.data() + roi_y * width + roi_x,                          \
                mask.data() + roi_y * width + roi_x, roi_height, roi_width,   \
                width, ref.data());                                           \
            ihist_hist##format_bits##_2d(                                     \
                sample_bits, data.data() + roi_y * width + roi_x,             \
                mask.data() + roi_y * width + roi_x, roi_height, roi_width,   \
                width, 1, 1, indices, hist.data(), parallel);                 \
        }                                                                     \
        CHECK(hist == ref);                                                   \
    }

#define TEST_CASE_ABC(format_bits, sample_bits)                               \
    TEST_CASE("abc" #format_bits "-" #sample_bits) {                          \
        auto const data = test_data<u##format_bits, sample_bits>(3 * size);   \
        auto const mask = test_data<u8, 1>(size);                             \
        constexpr std::size_t NBINS = 1 << sample_bits;                       \
        std::vector<u32> hist(3 * NBINS);                                     \
        std::vector<u32> ref(3 * NBINS);                                      \
        constexpr std::size_t indices[] = {0, 1, 2};                          \
                                                                              \
        bool const parallel = GENERATE(false, true);                          \
                                                                              \
        SECTION("nomask") {                                                   \
            ihist::histxy_unoptimized_st<u##format_bits, false, sample_bits,  \
                                         0, 3, 0, 1, 2>(                      \
                data.data() + 3 * (roi_y * width + roi_x), nullptr,           \
                roi_height, roi_width, width, ref.data());                    \
            ihist_hist##format_bits##_2d(                                     \
                sample_bits, data.data() + 3 * (roi_y * width + roi_x),       \
                nullptr, roi_height, roi_width, width, 3, 3, indices,         \
                hist.data(), parallel);                                       \
        }                                                                     \
        SECTION("mask") {                                                     \
            ihist::histxy_unoptimized_st<u##format_bits, true, sample_bits,   \
                                         0, 3, 0, 1, 2>(                      \
                data.data() + 3 * (roi_y * width + roi_x),                    \
                mask.data() + roi_y * width + roi_x, roi_height, roi_width,   \
                width, ref.data());                                           \
            ihist_hist##format_bits##_2d(                                     \
                sample_bits, data.data() + 3 * (roi_y * width + roi_x),       \
                mask.data() + roi_y * width + roi_x, roi_height, roi_width,   \
                width, 3, 3, indices, hist.data(), parallel);                 \
        }                                                                     \
        CHECK(hist == ref);                                                   \
    }

#define TEST_CASE_ABCX(format_bits, sample_bits)                              \
    TEST_CASE("abcx" #format_bits "-" #sample_bits) {                         \
        auto const data = test_data<u##format_bits, sample_bits>(4 * size);   \
        auto const mask = test_data<u8, 1>(size);                             \
        constexpr std::size_t NBINS = 1 << sample_bits;                       \
        std::vector<u32> hist(3 * NBINS);                                     \
        std::vector<u32> ref(3 * NBINS);                                      \
        constexpr std::size_t indices[] = {0, 1, 2};                          \
                                                                              \
        bool const parallel = GENERATE(false, true);                          \
                                                                              \
        SECTION("nomask") {                                                   \
            ihist::histxy_unoptimized_st<u##format_bits, false, sample_bits,  \
                                         0, 4, 0, 1, 2>(                      \
                data.data() + 4 * (roi_y * width + roi_x), nullptr,           \
                roi_height, roi_width, width, ref.data());                    \
            ihist_hist##format_bits##_2d(                                     \
                sample_bits, data.data() + 4 * (roi_y * width + roi_x),       \
                nullptr, roi_height, roi_width, width, 4, 3, indices,         \
                hist.data(), parallel);                                       \
        }                                                                     \
        SECTION("mask") {                                                     \
            ihist::histxy_unoptimized_st<u##format_bits, true, sample_bits,   \
                                         0, 4, 0, 1, 2>(                      \
                data.data() + 4 * (roi_y * width + roi_x),                    \
                mask.data() + roi_y * width + roi_x, roi_height, roi_width,   \
                width, ref.data());                                           \
            ihist_hist##format_bits##_2d(                                     \
                sample_bits, data.data() + 4 * (roi_y * width + roi_x),       \
                mask.data() + roi_y * width + roi_x, roi_height, roi_width,   \
                width, 4, 3, indices, hist.data(), parallel);                 \
        }                                                                     \
        CHECK(hist == ref);                                                   \
    }

#define TEST_CASE_XABC(format_bits, sample_bits)                              \
    TEST_CASE("xabc" #format_bits "-" #sample_bits) {                         \
        auto const data = test_data<u##format_bits, sample_bits>(4 * size);   \
        auto const mask = test_data<u8, 1>(size);                             \
        constexpr std::size_t NBINS = 1 << sample_bits;                       \
        std::vector<u32> hist(3 * NBINS);                                     \
        std::vector<u32> ref(3 * NBINS);                                      \
        constexpr std::size_t indices[] = {1, 2, 3};                          \
                                                                              \
        bool const parallel = GENERATE(false, true);                          \
                                                                              \
        SECTION("nomask") {                                                   \
            ihist::histxy_unoptimized_st<u##format_bits, false, sample_bits,  \
                                         0, 4, 1, 2, 3>(                      \
                data.data() + 4 * (roi_y * width + roi_x), nullptr,           \
                roi_height, roi_width, width, ref.data());                    \
            ihist_hist##format_bits##_2d(                                     \
                sample_bits, data.data() + 4 * (roi_y * width + roi_x),       \
                nullptr, roi_height, roi_width, width, 4, 3, indices,         \
                hist.data(), parallel);                                       \
        }                                                                     \
        SECTION("mask") {                                                     \
            ihist::histxy_unoptimized_st<u##format_bits, true, sample_bits,   \
                                         0, 4, 1, 2, 3>(                      \
                data.data() + 4 * (roi_y * width + roi_x),                    \
                mask.data() + roi_y * width + roi_x, roi_height, roi_width,   \
                width, ref.data());                                           \
            ihist_hist##format_bits##_2d(                                     \
                sample_bits, data.data() + 4 * (roi_y * width + roi_x),       \
                mask.data() + roi_y * width + roi_x, roi_height, roi_width,   \
                width, 4, 3, indices, hist.data(), parallel);                 \
        }                                                                     \
        CHECK(hist == ref);                                                   \
    }

#define TEST_CASE_DUAL(format_bits, sample_bits)                              \
    TEST_CASE("dual" #format_bits "-" #sample_bits) {                         \
        auto const data = test_data<u##format_bits, sample_bits>(2 * size);   \
        auto const mask = test_data<u8, 1>(size);                             \
        constexpr std::size_t NBINS = 1 << sample_bits;                       \
        std::vector<u32> hist(2 * NBINS);                                     \
        std::vector<u32> ref(2 * NBINS);                                      \
        constexpr std::size_t indices[] = {0, 1};                             \
                                                                              \
        bool const parallel = GENERATE(false, true);                          \
                                                                              \
        SECTION("nomask") {                                                   \
            ihist::histxy_unoptimized_st<u##format_bits, false, sample_bits,  \
                                         0, 2, 0, 1>(                         \
                data.data() + 2 * (roi_y * width + roi_x), nullptr,           \
                roi_height, roi_width, width, ref.data());                    \
            ihist_hist##format_bits##_2d(                                     \
                sample_bits, data.data() + 2 * (roi_y * width + roi_x),       \
                nullptr, roi_height, roi_width, width, 2, 2, indices,         \
                hist.data(), parallel);                                       \
        }                                                                     \
        SECTION("mask") {                                                     \
            ihist::histxy_unoptimized_st<u##format_bits, true, sample_bits,   \
                                         0, 2, 0, 1>(                         \
                data.data() + 2 * (roi_y * width + roi_x),                    \
                mask.data() + roi_y * width + roi_x, roi_height, roi_width,   \
                width, ref.data());                                           \
            ihist_hist##format_bits##_2d(                                     \
                sample_bits, data.data() + 2 * (roi_y * width + roi_x),       \
                mask.data() + roi_y * width + roi_x, roi_height, roi_width,   \
                width, 2, 2, indices, hist.data(), parallel);                 \
        }                                                                     \
        CHECK(hist == ref);                                                   \
    }

TEST_CASE_MONO(8, 8)
TEST_CASE_MONO(8, 5)
TEST_CASE_MONO(16, 16)
TEST_CASE_MONO(16, 15)
TEST_CASE_MONO(16, 12)
TEST_CASE_MONO(16, 11)
TEST_CASE_ABC(8, 8)
TEST_CASE_ABC(8, 5)
TEST_CASE_ABC(16, 16)
TEST_CASE_ABC(16, 15)
TEST_CASE_ABC(16, 12)
TEST_CASE_ABC(16, 11)
TEST_CASE_ABCX(8, 8)
TEST_CASE_ABCX(8, 5)
TEST_CASE_ABCX(16, 16)
TEST_CASE_ABCX(16, 15)
TEST_CASE_ABCX(16, 12)
TEST_CASE_ABCX(16, 11)
TEST_CASE_XABC(8, 8)
TEST_CASE_XABC(8, 5)
TEST_CASE_XABC(16, 16)
TEST_CASE_XABC(16, 15)
TEST_CASE_XABC(16, 12)
TEST_CASE_XABC(16, 11)
TEST_CASE_DUAL(8, 8)
TEST_CASE_DUAL(8, 5)
TEST_CASE_DUAL(16, 16)
TEST_CASE_DUAL(16, 15)
TEST_CASE_DUAL(16, 12)
TEST_CASE_DUAL(16, 11)