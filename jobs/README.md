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

## Sweeps

Most of the 125 archived scripts were not different experiments. Twenty-two of
them differ only in the value of `--pooling` or `--token-dir`, and some had
already been reduced to a `$POOL` variable in an attempt to stop writing new
ones. Vary the flag instead:

```sh
python jobs/submit.py --name aff --resource gpu_1 --hours 8 \
    --sweep pooling=mean,attn,meanmax -- \
    pipelines/train/scoring_head.py --pooling '{pooling}'
```

Each `{key}` in the command is replaced by that point's value, and one script is
written per point — `aff_pooling-mean.sh`, `aff_pooling-attn.sh`,
`aff_pooling-meanmax.sh`. Repeat `--sweep` for a cross product; the printed
billing estimate covers all of the jobs, not one.

Two things are refused rather than submitted: a `{key}` no `--sweep` defines (a
typo would otherwise reach the training script), and a `--sweep` nothing in the
command uses (which would queue N identical jobs and look like a comparison).
More than 24 jobs needs `--max-jobs` — a cross product is one line to write and
N node-hours to pay for.

Each point gets its own script rather than one array job indexing a shell array:
a wrong index is silent, and produces a plausible number with the wrong
hyper-parameter.

## Provenance

Job scripts are not tracked, so nothing in git says how a checkpoint was
produced. The run records that itself: every training run writes `run.json` into
its checkpoint directory with the command, the git SHA and whether the tree was
dirty, the seed, and — when it came from here — the job name, resource, id and
sweep point that `submit.py` passes through the environment.

The sweep point is worth recording even though its values are already in the
command: what the command does not say is that this run is one arm of a
comparison, and which axis was varied. That is what the old `job_pose_v8` /
`_v10` / `_v12` filenames were really encoding.

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
