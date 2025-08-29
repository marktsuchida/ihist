/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include <ihist.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

namespace ihist::internal {

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

} // namespace ihist::internal