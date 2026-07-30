# jobs/

Cluster job submission. Nothing here is tracked except the two files below —
the scripts actually submitted are site-specific and stay local.

```
lib.sh        the prologue a job sources: resolves the repo root, sets PYTHONPATH, exports $PY
submit.py     render a job script from a command; qsub it only on request
generated/    output of submit.py (git-ignored)
archive/      jobs previously run on TSUBAME, kept locally for provenance (git-ignored)
```

## Writing a job

```sh
python jobs/submit.py --name lm_pre --resource node_f --hours 8 \
    --description "ProLIT-CLM pretraining on the mixed corpus" \
    -- pipelines/train/clm.py --token-dir data/lm_tokens_pretrain_mixed
```

It writes the script and prints the `qsub` line with a billing estimate. It does
**not** submit unless you pass `--submit`: what a job does, which node it takes
and how long it may run should be agreed before it enters the queue.

Set `PROLIT_QSUB_GROUP` if your scheduler needs `qsub -g`.

## Provenance

Job scripts are not tracked, so nothing in git says how a checkpoint was
produced. The run records that itself: every training run writes `run.json` into
its checkpoint directory with the command, the git SHA and whether the tree was
dirty, the seed, and — when it came from here — the job name, resource and id
that `submit.py` passes through the environment.

Git has the code, the run directory has the weights and the command that made
them; together they reproduce the number. Delete the run and its provenance goes
with it, which is right: they are the same thing.

```sh
cat pocket-ligand-lm/<run>/checkpoints/run.json
```

`dirty: true` is the field to look at first — it means the code that produced
those weights was never committed, so the run cannot be reproduced from git
alone.

## Portability

`lib.sh` derives the repository root from its own location, so a generated
script runs from any directory and any checkout. Override `PROLIT_ROOT`, `PY` or
`WANDB_MODE` if you need something other than the defaults.

The resource names and billing coefficients in `submit.py` are TSUBAME's. On a
different scheduler, the pipelines under `pipelines/` are plain CLIs — run them
however that site expects; nothing in the library depends on this directory.

## The prologue trap

`lib.sh` exists mostly to stop one mistake from coming back. A job that began
with

```sh
source $HOME/.bashrc
module load cuda
```

exited after 0.3 s with status 0, no output, and 24 MB of vmem — python never
started. The torch wheel ships its own CUDA, so there is nothing to load, and
sourcing the interactive rc file in a non-interactive shell ends the script. The
failure is silent and looks like a successful run, which is what makes it worth
a file.
