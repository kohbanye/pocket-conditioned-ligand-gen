# Shared prologue for every TSUBAME job in this repository.
#
#   . "$(dirname "$0")/lib.sh"
#
# Sets the working directory, the import path and the offline W&B mode, and
# leaves $PY pointing at the interpreter a job should use.
#
# What this file deliberately does NOT do, because doing it kills the job:
#
#   source $HOME/.bashrc
#   module load cuda
#
# A job that ran those exited in 0.3 s with status 0, no output and 24 MB of
# vmem -- python never started. The torch wheel here ships its own CUDA, so
# there is nothing to load, and sourcing the interactive rc file in a
# non-interactive shell ends the script. Any new job script must use this
# prologue rather than reintroducing them.

PROLIT_ROOT="${PROLIT_ROOT:-/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen}"
cd "$PROLIT_ROOT" || exit 1

export PYTHONPATH="$PROLIT_ROOT/src:${PYTHONPATH}"
# Jobs run on compute nodes with no outbound network; sync the run afterwards.
export WANDB_MODE="${WANDB_MODE:-offline}"

# Call the venv interpreter directly. `uv run` re-resolves the editable install
# on every invocation, which is slow here and pointless inside a job that
# already has the environment.
PY="${PY:-$PROLIT_ROOT/.venv/bin/python}"
export PY
