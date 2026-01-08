#!/usr/bin/env bash
# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

#
# Check the staged JARs and POM (mainly for use by CI)
#

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <staging-directory>" >&2
    exit 1
fi

STAGING_DIR="$1"

if [[ ! -d "$STAGING_DIR" ]]; then
    echo "Error: Directory not found: $STAGING_DIR" >&2
    exit 1
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

errors=0

error() {
    echo "FAIL: $1" >&2
    ((errors++)) || true
}

pass() {
    echo "PASS: $1"
}

# Find version directory
IHIST_DIR="$STAGING_DIR/io/github/marktsuchida/ihist"
if [[ ! -d "$IHIST_DIR" ]]; then
    echo "Error: Directory not found: $IHIST_DIR" >&2
    exit 1
fi

mapfile -t version_dirs < <(find "$IHIST_DIR" -maxdepth 1 -mindepth 1 -type d)
if [[ ${#version_dirs[@]} -eq 0 ]]; then
    echo "Error: No version directory found under $IHIST_DIR" >&2
    exit 1
elif [[ ${#version_dirs[@]} -gt 1 ]]; then
    echo "Error: Multiple version directories found under $IHIST_DIR" >&2
    exit 1
fi

VERSION=$(basename "${version_dirs[0]}")
JAR_DIR="${version_dirs[0]}"
echo "Detected version: $VERSION"
echo

BARE_POM="$JAR_DIR/ihist-$VERSION.pom"
if [[ ! -f "$BARE_POM" ]]; then
    echo "Error: Expected POM not found: $BARE_POM" >&2
    exit 1
fi

# Expected files
MAIN_JAR="ihist-$VERSION.jar"
SOURCES_JAR="ihist-$VERSION-sources.jar"
JAVADOC_JAR="ihist-$VERSION-javadoc.jar"
NATIVES_LINUX_X86_64="ihist-$VERSION-natives-linux-x86_64.jar"
NATIVES_MACOS_X86_64="ihist-$VERSION-natives-macos-x86_64.jar"
NATIVES_MACOS_ARM64="ihist-$VERSION-natives-macos-arm64.jar"
NATIVES_WINDOWS_X86_64="ihist-$VERSION-natives-windows-x86_64.jar"

ALL_JARS=(
    "$MAIN_JAR"
    "$SOURCES_JAR"
    "$JAVADOC_JAR"
    "$NATIVES_LINUX_X86_64"
    "$NATIVES_MACOS_X86_64"
    "$NATIVES_MACOS_ARM64"
    "$NATIVES_WINDOWS_X86_64"
)

echo "=== Checking file presence ==="
if [[ -f "$BARE_POM" ]]; then
    pass "ihist-$VERSION.pom exists"
else
    error "ihist-$VERSION.pom is missing"
fi
for jar in "${ALL_JARS[@]}"; do
    if [[ -f "$JAR_DIR/$jar" ]]; then
        pass "$jar exists"
    else
        error "$jar is missing"
    fi
done
echo

echo "=== Checking for unrecognized files ==="
ALL_EXPECTED=("${ALL_JARS[@]}" "ihist-$VERSION.pom")
for file_path in "$JAR_DIR"/*; do
    [[ -f "$file_path" ]] || continue
    file=$(basename "$file_path")
    found=false
    for expected in "${ALL_EXPECTED[@]}"; do
        if [[ "$file" == "$expected" ]]; then
            found=true
            break
        fi
    done
    if ! $found; then
        error "unrecognized file: $file"
    fi
done
echo

echo "=== Checking embedded POMs ==="
POM_PATH="META-INF/maven/io.github.marktsuchida/ihist/pom.xml"
for jar in "${ALL_JARS[@]}"; do
    jar_path="$JAR_DIR/$jar"
    if [[ ! -f "$jar_path" ]]; then
        continue
    fi
    extracted_pom="$WORKDIR/pom-$(basename "$jar").xml"
    if unzip -p "$jar_path" "$POM_PATH" > "$extracted_pom" 2>/dev/null; then
        if diff -qw "$BARE_POM" "$extracted_pom" >/dev/null 2>&1; then
            pass "$jar: embedded POM matches bare POM"
        else
            error "$jar: embedded POM differs from bare POM"
        fi
    elif [[ "$jar" == "$JAVADOC_JAR" ]]; then
        pass "$jar: no embedded POM (expected for javadoc)"
    else
        error "$jar: no embedded POM found at $POM_PATH"
    fi
done
echo

echo "=== Checking native libraries ==="

check_native() {
    local jar="$1"
    local os="$2"
    local arch="$3"
    local lib_name="$4"
    shift 4
    local patterns=("$@")

    local jar_path="$JAR_DIR/$jar"
    if [[ ! -f "$jar_path" ]]; then
        return
    fi

    local lib_path="natives/$os/$arch/$lib_name"
    local extract_dir="$WORKDIR/native-$os-$arch"
    mkdir -p "$extract_dir"

    if ! unzip -j -o "$jar_path" "$lib_path" -d "$extract_dir" >/dev/null 2>&1
    then
        error "$jar: failed to extract $lib_path"
        return
    fi

    local extracted_lib="$extract_dir/$lib_name"
    if [[ ! -f "$extracted_lib" ]]; then
        error "$jar: $lib_path not found in JAR"
        return
    fi

    local file_output
    file_output=$(file "$extracted_lib")

    local all_matched=true
    for pattern in "${patterns[@]}"; do
        if ! echo "$file_output" | grep -qi "$pattern"; then
            all_matched=false
            break
        fi
    done

    if $all_matched; then
        pass "$jar: $lib_name has correct architecture ($os-$arch)"
    else
        error "$jar: $lib_name has wrong architecture: $file_output"
    fi
}

check_native "$NATIVES_LINUX_X86_64" "linux" "x86_64" "libihistj.so" \
    "ELF" "64-bit" "x86.64"
check_native "$NATIVES_MACOS_X86_64" "macos" "x86_64" "libihistj.dylib" \
    "Mach-O" "64-bit" "x86_64"
check_native "$NATIVES_MACOS_ARM64" "macos" "arm64" "libihistj.dylib" \
    "Mach-O" "64-bit" "arm64"
check_native "$NATIVES_WINDOWS_X86_64" "windows" "x86_64" "ihistj.dll" \
    "PE32+" "x86-64"
echo

if [[ $errors -gt 0 ]]; then
    echo "=== FAILED: $errors error(s) found ==="
    exit 1
else
    echo "=== ALL CHECKS PASSED ==="
    exit 0
fi
