# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

# Sharded build of explicit template instantiations, because otherwise it takes
# forever to build ihist_bench.
tmpl_libs = []
tmpl_shard_count = 15
foreach i : range(tmpl_shard_count)
    tmpl_libs += static_library(
        'tmpl_instantiations_' + i.to_string(),
        'tmpl_instantiations.cpp',
        include_directories: ihist_private_inc,
        cpp_args: [
            '-DIHIST_TMPL_SHARD=' + i.to_string(),
        ],
        dependencies: [
            ihist_dep,
        ],
    )
endforeach

tmpl_lib = static_library(
    'tmpl_instantiations',
    link_whole: tmpl_libs,
)

bench_deps = [
    ihist_dep,
    benchmark_dep,
]
if onetbb_dep.found()
    bench_deps += onetbb_dep
endif

bench_exe = executable(
    'ihist_bench',
    'ihist_bench.cpp',
    include_directories: ihist_private_inc,
    link_with: tmpl_lib,
    dependencies: bench_deps,
)
benchmark('ihist', bench_exe)

apibench_exe = executable(
    'api_bench',
    'api_bench.cpp',
    include_directories: ihist_private_inc,
    dependencies: [
        benchmark_dep,
        ihist_dep,
        opencv_dep,
    ],
)
benchmark('api', apibench_exe)