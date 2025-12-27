/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include "phys_core_count.hpp"

#include <catch2/catch_test_macros.hpp>

#include <thread>

namespace ihist::internal {

TEST_CASE("phys_core_count") {
    int const pcc = get_physical_core_count();
    CHECK(pcc > 0);
    int const lcc = std::thread::hardware_concurrency();
    CHECK(pcc <= lcc);
}

} // namespace ihist::internal
