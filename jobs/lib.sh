# Shared prologue for cluster jobs in this repository.
#
#   . "$(dirname "$0")/../lib.sh"
#
# Resolves the repository root from this file's own location, puts the library
# on the import path, and leaves $PY pointing at the interpreter to use. Nothing
# here is site-specific; override any of PROLIT_ROOT / PY / WANDB_MODE.
#
# What this file deliberately does NOT do, because doing it kills the job on
# TSUBAME:
#
#   source $HOME/.bashrc
#   module load cuda
#
# A job that ran those exited in 0.3 s with status 0, no output and 24 MB of
# vmem -- python never started. The torch wheel here ships its own CUDA, so
# there is nothing to load, and sourcing the interactive rc file in a
# non-interactive shell ends the script. Any new job script should use this
# prologue rather than reintroducing them.

_lib_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# jobs/generated/<job>.sh -> up two; jobs/<job>.sh -> up one.
case "$_lib_dir" in
    */jobs) PROLIT_ROOT="${PROLIT_ROOT:-$(dirname "$_lib_dir")}" ;;
    *)      PROLIT_ROOT="${PROLIT_ROOT:-$(dirname "$(dirname "$_lib_dir")")}" ;;
esac

cd "$PROLIT_ROOT" || exit 1
export PYTHONPATH="$PROLIT_ROOT/src:${PYTHONPATH}"

# Compute nodes usually have no outbound network; sync the run afterwards.
export WANDB_MODE="${WANDB_MODE:-offline}"

# Call the venv interpreter directly. `uv run` re-resolves the editable install
# on every invocation, which is slow and pointless inside a job that already has
# the environment.
PY="${PY:-$PROLIT_ROOT/.venv/bin/python}"
export PY
