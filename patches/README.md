# patches/

Local modifications to baseline sources under `third_party/`. Submodules are
pinned to upstream commits, so any edit made in a submodule working tree is
invisible to git and lost on a fresh clone — these patch files are the record.

Apply them after `git submodule update --init --recursive`:

```sh
sh scripts/apply_patches.sh          # all
sh scripts/apply_patches.sh DiffGui  # one baseline
```

The script is idempotent: an already-applied patch is skipped, not re-applied.

## DiffGui

`0001-make-internal-vina-docking-optional.patch` — DiffGui's sampler docks every
generated molecule inline with AutoDock Vina at `exhaustiveness=16`. The
benchmark re-docks all models uniformly afterwards, so that call is redundant,
very slow, and drags meeko / vina / AutoDockTools / pdb2pqr30 into the
generation environment. The patch makes the import and the call optional and
defaults `vina_score` to `0.0` on failure, so molecules are still emitted. It
does not touch the generative model or the sampling path, so generated
structures are unchanged.
