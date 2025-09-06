# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

benchmark_sources = files(
    'bench_mono10_mt.cpp',
    'bench_mono10_st.cpp',
    'bench_mono12_mt.cpp',
    'bench_mono12_st.cpp',
    'bench_mono14_mt.cpp',
    'bench_mono14_st.cpp',
    'bench_mono16_mt.cpp',
    'bench_mono16_st.cpp',
    'bench_mono8_mt.cpp',
    'bench_mono8_st.cpp',
    'bench_rgb10_mt.cpp',
    'bench_rgb10_st.cpp',
    'bench_rgb12_mt.cpp',
    'bench_rgb12_st.cpp',
    'bench_rgb14_mt.cpp',
    'bench_rgb14_st.cpp',
    'bench_rgb16_mt.cpp',
    'bench_rgb16_st.cpp',
    'bench_rgb8_mt.cpp',
    'bench_rgb8_st.cpp',
    'bench_rgbx10_mt.cpp',
    'bench_rgbx10_st.cpp',
    'bench_rgbx12_mt.cpp',
    'bench_rgbx12_st.cpp',
    'bench_rgbx14_mt.cpp',
    'bench_rgbx14_st.cpp',
    'bench_rgbx16_mt.cpp',
    'bench_rgbx16_st.cpp',
    'bench_rgbx8_mt.cpp',
    'bench_rgbx8_st.cpp',
    'ihist_bench.cpp',
)

bench_exe = executable(
    'ihist_bench',
    benchmark_sources,
    include_directories: ihist_private_inc,
    dependencies: [
        ihist_dep,
        benchmark_dep,
    ],
)
benchmark('ihist', bench_exe)

apibench_exe = executable(
    'api_bench',
    'api_bench.cpp',
    include_directories: ihist_private_inc,
    dependencies: [
        ihist_dep,
        benchmark_dep,
        opencv_dep,
    ],
)
benchmark('api', apibench_exe)