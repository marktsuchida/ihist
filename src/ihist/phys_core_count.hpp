/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#pragma once

namespace ihist::internal {

// Return the number of physical cores, or -1 if cannot determine.
// This should be equivalent to
// tbb::info::default_concurrency(tbb::task_arena::constraints{}.set_max_threads_per_core(1)),
// but that function only returns the correct value if TBB is built with
// TBBBind, which requires hwloc. The hwloc library (from which we could also
// directly get the physical core count) is inconvenient to build and to depend
// on, and it was easier to implement OS-specific code for this.
auto get_physical_core_count() -> int;

} // namespace ihist::internal