#!/bin/sh
# Build the OnionNet-2 scoring environment.
#
# TensorFlow comes from pip, not conda: conda-forge cannot solve tensorflow=2.3
# for any python it also carries, and the failure is a wall of
# "Could not solve for environment specs" rather than a missing-package message.
# OpenBabel stays on the conda side, where it works.
#
# protobuf==3.19.6 is load-bearing. TF 2.3 predates the protobuf 4.x C++
# descriptor change, and with a current protobuf `import tensorflow` dies with
# "Descriptors cannot not be created directly" -- which reads like a TF bug and
# is not one.
#
# numpy 1.18.5 / pandas 1.3.4 are upstream's own pins.
#
#   sh scripts/setup_onionnet2_env.sh
set -eu

BENCH_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ENV_DIR="${AFFINITY_ONIONNET2_ENV:-$BENCH_ROOT/.venv-onionnet2}"

micromamba create -y -p "$ENV_DIR" -c conda-forge python=3.8 openbabel

"$ENV_DIR/bin/pip" install \
    "tensorflow==2.3.1" "numpy==1.18.5" "pandas==1.3.4" \
    "protobuf==3.19.6" scikit-learn joblib scipy

"$ENV_DIR/bin/python" - <<'PY'
import tensorflow as tf, numpy, pandas  # noqa: F401
print("tf", tf.__version__, "| numpy", numpy.__version__, "| pandas", pandas.__version__)
print("OnionNet-2 environment OK")
PY

echo "Now fetch the model (git-lfs does not carry it on a plain checkout):"
echo "  python scripts/fetch_weights.py --onionnet2"
