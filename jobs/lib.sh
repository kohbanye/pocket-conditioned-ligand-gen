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

# Fail loudly when a GPU was paid for and is not there.
#
# torch falls back to CPU on its own, so a job that lands on a node where the
# device is not visible does not crash -- it runs the same work far slower,
# burns its whole walltime, and is killed at h_rt with a partial or absent
# result. That reads exactly like a slow job. Two ways in have been seen: the
# scheduler handing out a node whose GPU is already held, and a driver present
# but not usable from the job's cgroup.
#
# Gated on the resource type, and ONLY on the resource type. Guessing from the
# environment was tried and removed: $JOB_ID is set in an interactive session
# on this cluster too, so "are we inside a job" is not answerable here, and a
# check that guesses wrong kills a queued job for no reason. A script that does
# not export PROLIT_JOB_RESOURCE before sourcing this file simply gets no
# check -- which is what every script generated before that ordering was fixed
# gets, and is the same behaviour it had already.
#
# Set PROLIT_SKIP_GPU_CHECK=1 to run a GPU-typed job on CPU deliberately.
prolit_require_gpu() {
    [ "${PROLIT_SKIP_GPU_CHECK:-0}" = "1" ] && return 0
    case "${PROLIT_JOB_RESOURCE:-}" in
        node_f|node_h|node_q|node_o|gpu_1) ;;
        *) return 0 ;;
    esac
    if "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'
    then
        return 0
    fi
    echo "FATAL: ${PROLIT_JOB_RESOURCE} was requested but torch sees no CUDA" >&2
    echo "       device. Not falling back to CPU: that would burn the walltime" >&2
    echo "       and return a partial result that looks like a slow run." >&2
    nvidia-smi >&2 2>&1 || echo "       nvidia-smi unavailable" >&2
    exit 1
}

prolit_require_gpu
