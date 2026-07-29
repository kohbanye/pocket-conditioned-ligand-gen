#!/bin/sh
# Apply the local baseline patches under patches/ to the third_party/ submodules.
#
#   sh scripts/apply_patches.sh            # every baseline that has patches
#   sh scripts/apply_patches.sh DiffGui    # just one
#
# Idempotent: a patch that is already applied is reported and skipped.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
targets=${*:-$(ls "$root/patches" 2>/dev/null | grep -v '^README.md$' || true)}

if [ -z "$targets" ]; then
    echo "no patches to apply"
    exit 0
fi

status=0
for name in $targets; do
    dir="$root/patches/$name"
    sub="$root/third_party/$name"
    if [ ! -d "$dir" ]; then
        echo "!! no patch directory for '$name'" >&2
        status=1
        continue
    fi
    if [ ! -d "$sub" ]; then
        echo "!! submodule not checked out: third_party/$name" >&2
        echo "   run: git submodule update --init --recursive" >&2
        status=1
        continue
    fi
    for patch in "$dir"/*.patch; do
        [ -e "$patch" ] || continue
        label="$name/$(basename "$patch")"
        if git -C "$sub" apply --reverse --check "$patch" >/dev/null 2>&1; then
            echo "== $label: already applied, skipping"
        elif git -C "$sub" apply --check "$patch" >/dev/null 2>&1; then
            git -C "$sub" apply "$patch"
            echo "== $label: applied"
        else
            echo "!! $label: does not apply cleanly (upstream moved?)" >&2
            status=1
        fi
    done
done

exit "$status"
