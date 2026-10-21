#!/bin/sh
# Build the Sfcnn scoring environment.
#
# CPU only, deliberately: the CASF core set is 285 complexes, which needs no
# GPU, and a CPU build avoids matching a CUDA toolkit to TensorFlow.
#
# Two constraints are load-bearing:
#
#   * tensorflow < 2.16 -- 2.16 switched the bundled Keras to 3.x, whose
#     weight-loading path differs from the one `weights_22_112-0.0083.h5` was
#     written by. The pin is defensive rather than a reproduced failure: 2.15.0
#     is what this spec resolves to and is the version the measured row was
#     produced with, so it is the version to reproduce it.
#   * openbabel from conda-forge -- upstream's predict.py imports
#     `openbabel.pybel`, which the pip `openbabel` wheels do not provide. It
#     reads the protein PDB and ligand MOL2 directly, so there is no way around
#     it.
#
# Sfcnn needs no weight download: `weights_22_112-0.0083.h5` is an ordinary file
# in the repository, not tracked with git-lfs, so a plain checkout has the real
# thing rather than a pointer.
#
#   sh scripts/setup_sfcnn_env.sh
set -eu

BENCH_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ENV_DIR="${AFFINITY_SFCNN_ENV:-$BENCH_ROOT/.venv-sfcnn}"

micromamba create -y -p "$ENV_DIR" -c conda-forge \
    python=3.10 \
    "tensorflow>=2.10,<2.16" \
    openbabel \
    numpy \
    pandas \
    scikit-learn

# Fail here rather than deep inside Keras or a missing-module traceback later.
"$ENV_DIR/bin/python" - <<'PY'
import tensorflow as tf
from openbabel import pybel  # noqa: F401  - import is the check

print("tensorflow", tf.__version__)
print("openbabel.pybel import ok")
PY

echo
echo "Sfcnn environment ready: $ENV_DIR"
echo "Point the adapter at it with AFFINITY_SFCNN_PYTHON if it lives elsewhere."
