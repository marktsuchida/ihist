/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

// Consumers of ihist should define IHIST_SHARED when linking to a shared
// library build of ihist. But it is optional (in theory, slightly better
// performance on Windows; probably negligible in practice).

#if defined _WIN32 || defined __CYGWIN__
#if defined(IHIST_BUILDING_SHARED)
#define IHIST_PUBLIC __declspec(dllexport)
#elif defined(IHIST_SHARED)
#define IHIST_PUBLIC __declspec(dllimport)
#endif
#elif defined(IHIST_BUILDING_SHARED)
#define IHIST_PUBLIC __attribute__((visibility("default")))
#endif

#ifndef IHIST_PUBLIC
#define IHIST_PUBLIC
#endif

#ifdef __cplusplus
extern "C" {

#ifdef _MSC_VER
#define IHIST_RESTRICT __restrict
#else
#define IHIST_RESTRICT __restrict__
#endif
#endif

#ifndef IHIST_RESTRICT
#define IHIST_RESTRICT restrict
#endif

IHIST_PUBLIC void
ihist_hist8_2d(size_t sample_bits, uint8_t const *IHIST_RESTRICT image,
               uint8_t const *IHIST_RESTRICT mask, size_t height, size_t width,
               size_t image_stride, size_t mask_stride, size_t n_components,
               size_t n_hist_components,
               size_t const *IHIST_RESTRICT component_indices,
               uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel);

IHIST_PUBLIC void
ihist_hist16_2d(size_t sample_bits, uint16_t const *IHIST_RESTRICT image,
                uint8_t const *IHIST_RESTRICT mask, size_t height,
                size_t width, size_t image_stride, size_t mask_stride,
                size_t n_components, size_t n_hist_components,
                size_t const *IHIST_RESTRICT component_indices,
                uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel);

#ifdef __cplusplus
} // extern "C"
#endif