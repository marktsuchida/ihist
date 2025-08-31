/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined _WIN32 || defined __CYGWIN__
#ifdef IHIST_BUILDING_SHARED
#define IHIST_PUBLIC __declspec(dllexport)
#else
#define IHIST_PUBLIC __declspec(dllimport)
#endif
#else
#ifdef IHIST_BUILDING_SHARED
#define IHIST_PUBLIC __attribute__((visibility("default")))
#else
#define IHIST_PUBLIC
#endif
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
ihist_hist8_mono_2d(size_t sample_bits, uint8_t const *IHIST_RESTRICT image,
                    uint8_t const *IHIST_RESTRICT mask, size_t width,
                    size_t height, size_t roi_x, size_t roi_y,
                    size_t roi_width, size_t roi_height,
                    uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel);
IHIST_PUBLIC void
ihist_hist8_abc_2d(size_t sample_bits, uint8_t const *IHIST_RESTRICT image,
                   uint8_t const *IHIST_RESTRICT mask, size_t width,
                   size_t height, size_t roi_x, size_t roi_y, size_t roi_width,
                   size_t roi_height, uint32_t *IHIST_RESTRICT histogram,
                   bool maybe_parallel);
IHIST_PUBLIC void
ihist_hist8_abcx_2d(size_t sample_bits, uint8_t const *IHIST_RESTRICT image,
                    uint8_t const *IHIST_RESTRICT mask, size_t width,
                    size_t height, size_t roi_x, size_t roi_y,
                    size_t roi_width, size_t roi_height,
                    uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel);
IHIST_PUBLIC void
ihist_hist8_xabc_2d(size_t sample_bits, uint8_t const *IHIST_RESTRICT image,
                    uint8_t const *IHIST_RESTRICT mask, size_t width,
                    size_t height, size_t roi_x, size_t roi_y,
                    size_t roi_width, size_t roi_height,
                    uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel);
IHIST_PUBLIC void
ihist_hist16_mono_2d(size_t sample_bits, uint16_t const *IHIST_RESTRICT image,
                     uint8_t const *IHIST_RESTRICT mask, size_t width,
                     size_t height, size_t roi_x, size_t roi_y,
                     size_t roi_width, size_t roi_height,
                     uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel);
IHIST_PUBLIC void
ihist_hist16_abc_2d(size_t sample_bits, uint16_t const *IHIST_RESTRICT image,
                    uint8_t const *IHIST_RESTRICT mask, size_t width,
                    size_t height, size_t roi_x, size_t roi_y,
                    size_t roi_width, size_t roi_height,
                    uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel);
IHIST_PUBLIC void
ihist_hist16_abcx_2d(size_t sample_bits, uint16_t const *IHIST_RESTRICT image,
                     uint8_t const *IHIST_RESTRICT mask, size_t width,
                     size_t height, size_t roi_x, size_t roi_y,
                     size_t roi_width, size_t roi_height,
                     uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel);
IHIST_PUBLIC void
ihist_hist16_xabc_2d(size_t sample_bits, uint16_t const *IHIST_RESTRICT image,
                     uint8_t const *IHIST_RESTRICT mask, size_t width,
                     size_t height, size_t roi_x, size_t roi_y,
                     size_t roi_width, size_t roi_height,
                     uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel);

#ifdef __cplusplus
} // extern "C"
#endif