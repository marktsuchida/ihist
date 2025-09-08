# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

bench_exe = executable(
    'ihist_bench',
    'ihist_bench.cpp',
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