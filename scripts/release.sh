#!/usr/bin/env bash
# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

set -euo pipefail

usage() {
    cat <<EOF >&2
Usage: $0 [OPTIONS]

Create a release by updating version, committing, and tagging.

Options:
  --push          Push commits and tag to remote (default: print commands only)
  --dry-run       Show what would be done without making changes
  --remote=NAME   Use specified remote (default: origin)
  -h, --help      Show this help message

The script will:
1. Validate that main is up-to-date with the remote
2. Update meson.build version from X.Y.Z.dev0 to X.Y.Z
3. Commit "Release X.Y.Z" and tag vX.Y.Z
4. Update meson.build version to X.Y.(Z+1).dev0
5. Commit "Version back to dev"
EOF
    exit 1
}

error() {
    echo "Error: $1" >&2
    exit 1
}

info() {
    echo ":: $1"
}

PUSH=false
DRY_RUN=false
REMOTE="origin"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)
            PUSH=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --remote=*)
            REMOTE="${1#--remote=}"
            if [[ -z "$REMOTE" ]]; then
                echo "Error: --remote requires a value" >&2
                usage
            fi
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

for cmd in uvx jq git; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        error "Required command '$cmd' not found in PATH"
    fi
done

info "Checking branch"
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    error "Must be on main branch (currently on '$CURRENT_BRANCH')"
fi

info "Checking working directory"
if [[ -n "$(git status --porcelain)" ]]; then
    error "Working directory has uncommitted changes"
fi

info "Validating remote '$REMOTE'"
REMOTE_URL=$(git remote get-url "$REMOTE" 2>/dev/null) || \
    error "Remote '$REMOTE' not found"
if [[ "$REMOTE_URL" != *marktsuchida/ihist* ]]; then
    error "Remote URL '$REMOTE_URL' does not match marktsuchida/ihist"
fi

info "Fetching from $REMOTE"
git fetch "$REMOTE"

info "Checking sync with remote"
LOCAL_HEAD=$(git rev-parse HEAD)
REMOTE_HEAD=$(git rev-parse "$REMOTE/main")
if [[ "$LOCAL_HEAD" != "$REMOTE_HEAD" ]]; then
    error "Local main is not in sync with $REMOTE/main"
fi

info "Computing versions"
CURRENT=$(uvx meson rewrite kwargs info project / | jq -r '.kwargs["project#/"].version') || \
    error "Failed to read version from meson.build"
if [[ -z "$CURRENT" || "$CURRENT" == "null" ]]; then
    error "Failed to extract version from meson.build"
fi
if [[ "$CURRENT" != *.dev0 ]]; then
    error "Current version '$CURRENT' does not end with .dev0"
fi

RELEASE=${CURRENT%.dev0}
IFS='.' read -r MAJOR MINOR PATCH <<< "$RELEASE"
if [[ -z "$MAJOR" || -z "$MINOR" || -z "$PATCH" ]]; then
    error "Failed to parse version '$RELEASE' as MAJOR.MINOR.PATCH"
fi
if ! [[ "$PATCH" =~ ^[0-9]+$ ]]; then
    error "PATCH version '$PATCH' is not a number"
fi
NEXT="$MAJOR.$MINOR.$((PATCH + 1)).dev0"

info "Versions: $CURRENT -> $RELEASE -> $NEXT"

TAG="v$RELEASE"
info "Checking tag '$TAG'"
if git rev-parse "$TAG" >/dev/null 2>&1; then
    error "Tag '$TAG' already exists locally"
fi
if git ls-remote --tags "$REMOTE" "$TAG" | grep -q .; then
    error "Tag '$TAG' already exists on remote"
fi

if $DRY_RUN; then
    echo
    echo "=== Dry run - would execute: ==="
    echo "uvx meson rewrite kwargs set project / version '$RELEASE'"
    echo "git add meson.build"
    echo "git commit -m 'Release $RELEASE'"
    echo "git tag -a '$TAG' -m 'Release'"
    echo "uvx meson rewrite kwargs set project / version '$NEXT'"
    echo "git add meson.build"
    echo "git commit -m 'Version back to dev'"
    if $PUSH; then
        echo "git push $REMOTE HEAD"
        echo "git push $REMOTE '$TAG'"
    else
        echo
        echo "=== Would print push commands (--push not specified) ==="
    fi
    exit 0
fi

info "Setting version to $RELEASE"
uvx meson rewrite kwargs set project / version "$RELEASE"

info "Creating release commit"
git add meson.build
git commit -m "Release $RELEASE"

info "Creating tag $TAG"
git tag -a "$TAG" -m "Release"

info "Setting version to $NEXT"
uvx meson rewrite kwargs set project / version "$NEXT"

info "Creating dev version commit"
git add meson.build
git commit -m "Version back to dev"

if $PUSH; then
    info "Pushing to $REMOTE"
    git push "$REMOTE" HEAD
    if ! git push "$REMOTE" "$TAG"; then
        echo "Error: Commits pushed but tag push failed!" >&2
        echo "To push the tag manually: git push $REMOTE '$TAG'" >&2
        exit 1
    fi
    info "Done - release $RELEASE pushed"
else
    echo
    echo "=== Release created locally ==="
    echo "To push, run:"
    echo "  git push $REMOTE HEAD"
    echo "  git push $REMOTE '$TAG'"
fi
