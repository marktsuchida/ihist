# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

test_srcs = files(
    'test_accumulation.cpp',
    'test_bin_mapping.cpp',
    'test_components.cpp',
    'test_core_count.cpp',
    'test_edge_cases.cpp',
    'test_implementation_variants.cpp',
    'test_region_selection.cpp',
)

test_exe = executable(
    'ihist_test',
    test_srcs,
    include_directories: ihist_private_inc,
    dependencies: [
        ihist_dep,
        catch2_with_main_dep,
    ],
)

test_shard_count = 16
foreach i : range(test_shard_count)
    test(
        'ihist-' + i.to_string(),
        test_exe,
        args: [
            '--shard-count=' + test_shard_count.to_string(),
            '--shard-index=' + i.to_string(),
        ],
        timeout: 300, # Debug build is very slow.
    )
endforeach