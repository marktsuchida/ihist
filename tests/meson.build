# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

test_srcs = files(
    'all_values.cpp',
    'api.cpp',
    'constant_input.cpp',
    'empty_input.cpp',
    'internal_utils.cpp',
    'random_input.cpp',
)

# We do not use ihist_lib here, because we also directly call ihist internals.
# If we link ihist as a library, oneTBB templates get instantiated in different
# ways (different visibility?) and fail to merge at link time, causing
# mysterious failures (macOS, unoptimized builds).
test_exe = executable(
    'ihist_test',
    [
        ihist_srcs,
        test_srcs,
    ],
    include_directories: [
        ihist_private_inc,
        ihist_public_inc,
    ],
    dependencies: [
        onetbb_dep,
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