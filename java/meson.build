# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

# Find Java compiler
javac = find_program('javac', required: true)

# Get JAVA_HOME from environment or derive from javac location
java_home = run_command(
    'sh', '-c',
    '''
    if [ -n "$JAVA_HOME" ]; then
        echo "$JAVA_HOME"
    else
        # Derive from javac location
        javac_path=$(command -v javac)
        if [ -L "$javac_path" ]; then
            javac_path=$(readlink -f "$javac_path" 2>/dev/null || realpath "$javac_path" 2>/dev/null || echo "$javac_path")
        fi
        echo "$(dirname "$(dirname "$javac_path")")"
    fi
    ''',
    check: true,
).stdout().strip()

message('Using JAVA_HOME: ' + java_home)

# JNI include directories
jni_inc_base = java_home / 'include'

if host_machine.system() == 'darwin'
    jni_inc_platform = jni_inc_base / 'darwin'
elif host_machine.system() == 'windows'
    jni_inc_platform = jni_inc_base / 'win32'
else
    jni_inc_platform = jni_inc_base / 'linux'
endif

# Use compiler-specific system include flags to avoid Meson's rejection of
# absolute paths that may be inside the source tree.
cxx = meson.get_compiler('cpp')
if cxx.get_argument_syntax() == 'msvc'
    jni_args = ['/external:I', jni_inc_base, '/external:I', jni_inc_platform]
else
    jni_args = ['-isystem', jni_inc_base, '-isystem', jni_inc_platform]
endif
jni_inc_dep = declare_dependency(compile_args: jni_args)

# Java source files for JNI header generation
java_native_src = files('src/main/java/ihistj/IHistNative.java')

# Directory for generated JNI headers
jni_header_dir = meson.current_build_dir() / 'jni-headers'

# Directory for class files output
jni_classes_dir = meson.current_build_dir() / 'classes'

# Generate JNI header using javac -h
# This compiles the Java class and generates the native header
# We use sh to create directories and copy the header to the expected output location
jni_header = custom_target(
    'jni-header',
    input: java_native_src,
    output: 'ihistj_IHistNative.h',
    command: [
        'sh', '-c',
        'mkdir -p "$1" "$2" && "$3" -h "$1" -d "$2" -source 8 -target 8 "$4" && cp "$1/ihistj_IHistNative.h" "$5"',
        '--',
        jni_header_dir,
        jni_classes_dir,
        javac,
        '@INPUT@',
        '@OUTPUT@',
    ],
)

# Include directory for generated JNI header
jni_generated_inc = include_directories('.')

# Hide symbols from statically linked dependencies (e.g., TBB)
symbol_hide_link_args = []
symbol_hide_link_depends = []

if host_machine.system() == 'linux'
    symbol_hide_link_args += ['-Wl,--exclude-libs,ALL']
elif host_machine.system() == 'darwin'
    # For macOS, we could use an exported symbols list like Python bindings,
    # but for JNI we typically just need the JNI_OnLoad and Java_* symbols.
    # The gnu_symbol_visibility: 'hidden' handles most of it, and JNI functions
    # are explicitly exported via JNIEXPORT macro.
endif

# Build the native library
ihistj_lib = shared_library(
    'ihistj',
    'src/main/cpp/ihistj_jni.cpp',
    jni_header,
    include_directories: [jni_generated_inc],
    dependencies: [
        ihist_dep,
        jni_inc_dep,
    ],
    cpp_args: ['-I' + jni_header_dir],
    link_args: symbol_hide_link_args,
    link_depends: symbol_hide_link_depends,
    gnu_symbol_visibility: 'hidden',
    install: false,  # Don't install for now
)
