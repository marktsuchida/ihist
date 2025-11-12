/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "ihist.hpp"

#ifndef IHIST_TMPL_EXTERN_0
#define IHIST_TMPL_EXTERN_0 extern
#endif
#ifndef IHIST_TMPL_EXTERN_1
#define IHIST_TMPL_EXTERN_1 extern
#endif
#ifndef IHIST_TMPL_EXTERN_2
#define IHIST_TMPL_EXTERN_2 extern
#endif
#ifndef IHIST_TMPL_EXTERN_3
#define IHIST_TMPL_EXTERN_3 extern
#endif
#ifndef IHIST_TMPL_EXTERN_4
#define IHIST_TMPL_EXTERN_4 extern
#endif
#ifndef IHIST_TMPL_EXTERN_5
#define IHIST_TMPL_EXTERN_5 extern
#endif
#ifndef IHIST_TMPL_EXTERN_6
#define IHIST_TMPL_EXTERN_6 extern
#endif
#ifndef IHIST_TMPL_EXTERN_7
#define IHIST_TMPL_EXTERN_7 extern
#endif
#ifndef IHIST_TMPL_EXTERN_8
#define IHIST_TMPL_EXTERN_8 extern
#endif
#ifndef IHIST_TMPL_EXTERN_9
#define IHIST_TMPL_EXTERN_9 extern
#endif
#ifndef IHIST_TMPL_EXTERN_10
#define IHIST_TMPL_EXTERN_10 extern
#endif
#ifndef IHIST_TMPL_EXTERN_11
#define IHIST_TMPL_EXTERN_11 extern
#endif
#ifndef IHIST_TMPL_EXTERN_12
#define IHIST_TMPL_EXTERN_12 extern
#endif
#ifndef IHIST_TMPL_EXTERN_13
#define IHIST_TMPL_EXTERN_13 extern
#endif
#ifndef IHIST_TMPL_EXTERN_14
#define IHIST_TMPL_EXTERN_14 extern
#endif

namespace ihist {

template <std::size_t Stripes, std::size_t Unrolls>
constexpr tuning_parameters tuning{Stripes, Unrolls};

#define IHIST_TMPL_INST_1D(mt, T, mask, bits, stripes, unrolls, n_components, \
                           ...)                                               \
    template void hist_striped_##mt<tuning<stripes, unrolls>, T, mask, bits,  \
                                    0, n_components, __VA_ARGS__>(            \
        T const *IHIST_RESTRICT, std::uint8_t const *IHIST_RESTRICT,          \
        std::size_t, std::uint32_t *IHIST_RESTRICT, std::size_t);

#define IHIST_TMPL_INST_2D(mt, T, mask, bits, stripes, unrolls, n_components, \
                           ...)                                               \
    template void histxy_striped_##mt<tuning<stripes, unrolls>, T, mask,      \
                                      bits, 0, n_components, __VA_ARGS__>(    \
        T const *IHIST_RESTRICT, std::uint8_t const *IHIST_RESTRICT,          \
        std::size_t, std::size_t, std::size_t, std::uint32_t *IHIST_RESTRICT, \
        std::size_t);

#define IHIST_TMPL_INST_BOTHD(Extern, mt, T, mask, bits, stripes, unrolls,    \
                              n_components, ...)                              \
    Extern IHIST_TMPL_INST_1D(mt, T, mask, bits, stripes, unrolls,            \
                              n_components, __VA_ARGS__)                      \
    Extern IHIST_TMPL_INST_2D(mt, T, mask, bits, stripes, unrolls,            \
                              n_components, __VA_ARGS__)

#define IHIST_TMPL_INST_MT(Extern, T, mask, bits, stripes, unrolls,           \
                           n_components, ...)                                 \
    IHIST_TMPL_INST_BOTHD(Extern, st, T, mask, bits, stripes, unrolls,        \
                          n_components, __VA_ARGS__)                          \
    IHIST_TMPL_INST_BOTHD(Extern, mt, T, mask, bits, stripes, unrolls,        \
                          n_components, __VA_ARGS__)

#define IHIST_TMPL_INST_BITS(Extern, mask, stripes, unrolls, n_components,    \
                             ...)                                             \
    IHIST_TMPL_INST_MT(Extern, std::uint8_t, mask, 8, stripes, unrolls,       \
                       n_components, __VA_ARGS__)                             \
    IHIST_TMPL_INST_MT(Extern, std::uint16_t, mask, 10, stripes, unrolls,     \
                       n_components, __VA_ARGS__)                             \
    IHIST_TMPL_INST_MT(Extern, std::uint16_t, mask, 12, stripes, unrolls,     \
                       n_components, __VA_ARGS__)                             \
    IHIST_TMPL_INST_MT(Extern, std::uint16_t, mask, 14, stripes, unrolls,     \
                       n_components, __VA_ARGS__)                             \
    IHIST_TMPL_INST_MT(Extern, std::uint16_t, mask, 16, stripes, unrolls,     \
                       n_components, __VA_ARGS__)

#define IHIST_TMPL_INST_MASK(Extern, stripes, unrolls, n_components, ...)     \
    IHIST_TMPL_INST_BITS(Extern, false, stripes, unrolls, n_components,       \
                         __VA_ARGS__)                                         \
    IHIST_TMPL_INST_BITS(Extern, true, stripes, unrolls, n_components,        \
                         __VA_ARGS__)

#define IHIST_TMPL_INST_STRIPES(Extern, unrolls, n_components, ...)           \
    IHIST_TMPL_INST_MASK(Extern, 1, unrolls, n_components, __VA_ARGS__)       \
    IHIST_TMPL_INST_MASK(Extern, 2, unrolls, n_components, __VA_ARGS__)       \
    IHIST_TMPL_INST_MASK(Extern, 4, unrolls, n_components, __VA_ARGS__)       \
    IHIST_TMPL_INST_MASK(Extern, 8, unrolls, n_components, __VA_ARGS__)       \
    IHIST_TMPL_INST_MASK(Extern, 16, unrolls, n_components, __VA_ARGS__)

#define IHIST_TMPL_INST_ALL()                                                 \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_0, 1, 1, 0)                     \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_1, 2, 1, 0)                     \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_2, 4, 1, 0)                     \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_3, 8, 1, 0)                     \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_4, 16, 1, 0)                    \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_5, 1, 3, 0, 1, 2)               \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_6, 2, 3, 0, 1, 2)               \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_7, 4, 3, 0, 1, 2)               \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_8, 8, 3, 0, 1, 2)               \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_9, 16, 3, 0, 1, 2)              \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_10, 1, 4, 0, 1, 2)              \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_11, 2, 4, 0, 1, 2)              \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_12, 4, 4, 0, 1, 2)              \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_13, 8, 4, 0, 1, 2)              \
    IHIST_TMPL_INST_STRIPES(IHIST_TMPL_EXTERN_14, 16, 4, 0, 1, 2)

IHIST_TMPL_INST_ALL()

} // namespace ihist