/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "ihist/ihist.h"

namespace ihist::internal {

// Return the number of physical cores, or -1 if cannot determine.
// This should be equivalent to
// tbb::info::default_concurrency(tbb::task_arena::constraints{}.set_max_threads_per_core(1)),
// but that function only returns the correct value if TBB is built with
// TBBBind, which requires hwloc. The hwloc library (from which we could also
// directly get the physical core count) is inconvenient to build and to depend
// on, and it was easier to implement OS-specific code for this.

// This function is marked public even though it is not part of the API, for
// two reasons:
// - It's needed for tests to link when we are configured to build ihist as a
//   shared library.
// - It may come in handy for troubleshooting purposes in user code.

IHIST_PUBLIC auto get_physical_core_count() -> int;

} // namespace ihist::internal