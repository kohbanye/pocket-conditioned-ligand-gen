#!/bin/bash
# Build the gnina container this bench scores with.
#
#   scripts/fetch_gnina.sh [output.sif]
#
# Why a container and not the release binary: gnina publishes one asset,
# `gnina.cuda12.8.static`, and despite the name it is dynamically linked against
# glibc 2.35 and libstdc++ GLIBCXX_3.4.30. This cluster is RHEL 9 (glibc 2.34),
# so the binary will not start:
#
#   gnina: /lib64/libm.so.6: version `GLIBC_2.35' not found
#
# The official image carries its own userland, and its digest pins the build --
# which a locally compiled gnina would not.
#
# This is large: the image unpacks to tens of GB before it is squashed into the
# .sif, so both the cache and the scratch directory are pointed at the group
# filesystem rather than $HOME, which has neither the space nor the inodes.
set -euo pipefail

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
bench=$(dirname "$here")
out=${1:-$bench/weights/gnina/gnina.sif}

scratch=${GNINA_BUILD_DIR:-${TMPDIR:-/tmp}/gnina-build-$USER}
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$scratch/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$scratch/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$out")"

echo "building $out"
echo "  cache   $APPTAINER_CACHEDIR"
echo "  scratch $APPTAINER_TMPDIR"
apptainer pull --force "$out" "docker://${GNINA_IMAGE:-gnina/gnina:latest}"

echo "verifying"
apptainer exec "$out" gnina --version

cat <<EOF

done: $out

The bench finds it at that path on its own. To use a different build, set
RESCORING_BENCH_GNINA_SIF to the .sif, or RESCORING_BENCH_GNINA to a complete
command that runs gnina.

The scratch directory above can now be removed.
EOF
