# jobs/

TSUBAME (SGE) job submission.

```
lib.sh        the prologue every job sources
submit.py     generate a job script from a command; qsub it only on request
generated/    output of submit.py (git-ignored)
archive/      the 125 scripts as actually run, kept for provenance
```

## Writing a new job

```sh
python jobs/submit.py --name lm_pre --resource node_f --hours 8 \
    --description "ProLIT-CLM pretraining on the mixed corpus" \
    -- pipelines/train/clm.py --token-dir data/lm_tokens_pretrain_mixed
```

It writes the script and prints the `qsub` line plus a billing estimate. It does
**not** submit unless you pass `--submit`: what a job does, which node it takes
and how long it may run should be agreed before it enters the queue.

## The prologue trap

`lib.sh` exists mostly to stop one mistake from coming back. A job that began
with

```sh
source $HOME/.bashrc
module load cuda
```

exited after 0.3 s with status 0, no output, and 24 MB of vmem — python never
started. The torch wheel here ships its own CUDA, so there is nothing to load,
and sourcing the interactive rc file in a non-interactive shell ends the script.
The failure is silent and looks like a successful run, which is what makes it
worth a file. Source `lib.sh` instead.

## archive/

125 scripts, one per experiment arm actually submitted: the tokenizer ablation
(`_joint` / `_sep` / `_sep4096` across every stage), the rescoring-head sweep
(`job_pose_v*`), the decoy-corpus generations, the CASF evaluations. They name
specific runs and checkpoints, so they are the record of how the paper's
artifacts were produced — read them for provenance, not as templates. New work
should go through `submit.py`.

Their entry-point paths were updated when `scripts/` was split into
`pipelines/`, so they still point at real files, but they have not been re-run
since.
