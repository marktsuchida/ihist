/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#ifndef IHIST_TMPL_SHARD
#error IHIST_TMPL_SHARD must be defined
#endif

#if IHIST_TMPL_SHARD == 0
#define IHIST_TMPL_EXTERN_0
#elif IHIST_TMPL_SHARD == 1
#define IHIST_TMPL_EXTERN_1
#elif IHIST_TMPL_SHARD == 2
#define IHIST_TMPL_EXTERN_2
#elif IHIST_TMPL_SHARD == 3
#define IHIST_TMPL_EXTERN_3
#elif IHIST_TMPL_SHARD == 4
#define IHIST_TMPL_EXTERN_4
#elif IHIST_TMPL_SHARD == 5
#define IHIST_TMPL_EXTERN_5
#elif IHIST_TMPL_SHARD == 6
#define IHIST_TMPL_EXTERN_6
#elif IHIST_TMPL_SHARD == 7
#define IHIST_TMPL_EXTERN_7
#elif IHIST_TMPL_SHARD == 8
#define IHIST_TMPL_EXTERN_8
#elif IHIST_TMPL_SHARD == 9
#define IHIST_TMPL_EXTERN_9
#elif IHIST_TMPL_SHARD == 10
#define IHIST_TMPL_EXTERN_10
#elif IHIST_TMPL_SHARD == 11
#define IHIST_TMPL_EXTERN_11
#elif IHIST_TMPL_SHARD == 12
#define IHIST_TMPL_EXTERN_12
#elif IHIST_TMPL_SHARD == 13
#define IHIST_TMPL_EXTERN_13
#elif IHIST_TMPL_SHARD == 14
#define IHIST_TMPL_EXTERN_14
#else
#error Invalid shard (IHIST_TMPL_SHARD must be 0-14)
#endif

#include "tmpl_instantiations.hpp"