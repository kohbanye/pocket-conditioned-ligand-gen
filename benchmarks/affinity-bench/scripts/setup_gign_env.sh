#!/bin/sh
# Build the GIGN scoring environment.
#
# CPU only, deliberately: the CASF core set is 285 complexes, which needs no
# GPU, and a CPU build avoids matching a CUDA toolkit to torch 1.10.
#
# Three pins are load-bearing and each one cost a debugging cycle:
#
#   * mkl<2024 -- torch 1.10 expects the symbol iJIT_NotifyEvent, which newer
#     MKL removed. Without this, `import torch` fails with
#     "libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent".
#   * torch_scatter==2.0.9 / torch_sparse==0.6.13 -- torch_geometric 2.0.3
#     hard-imports torch_sparse. Unpinned, pip compiles a current version that
#     calls torch.sparse_csc_tensor, which torch 1.10 does not have, and the
#     failure appears at import time rather than at install time.
#   * rdkit unpinned -- upstream asks for 2022.09.2, which conda-forge does not
#     carry. The featurisation uses stable APIs, so a current rdkit is fine.
#
#   sh scripts/setup_gign_env.sh
set -eu

BENCH_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ENV_DIR="${AFFINITY_GIGN_ENV:-$BENCH_ROOT/.venv-gign}"

micromamba create -y -p "$ENV_DIR" -c pytorch -c conda-forge \
    python=3.9 pytorch=1.10.2 cpuonly \
    rdkit pymol-open-source openbabel \
    pandas scipy scikit-learn networkx tqdm

# pymol carves the 5 A pocket; obabel converts the ligand mol2 to pdb. Both are
# called by GIGN's own preprocessing, so both must be in this environment.
micromamba install -y -p "$ENV_DIR" -c conda-forge "mkl<2024"

"$ENV_DIR/bin/pip" install "torch_geometric==2.0.3"
"$ENV_DIR/bin/pip" install "torch_scatter==2.0.9" "torch_sparse==0.6.13" \
    -f https://data.pyg.org/whl/torch-1.10.2+cpu.html

"$ENV_DIR/bin/python" - <<'PY'
import torch, torch_geometric, torch_sparse, torch_scatter, rdkit, pymol  # noqa: F401
print("torch", torch.__version__, "| pyg", torch_geometric.__version__,
      "| sparse", torch_sparse.__version__, "| scatter", torch_scatter.__version__)
print("GIGN environment OK")
PY
