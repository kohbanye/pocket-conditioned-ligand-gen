# pipelines/

Everything that turns raw structures into trained weights. These are argparse
CLIs, not a library: they read from disk, write to disk, and are what the job
scripts under `jobs/` invoke.

```
corpora/    raw structures -> descriptor cache -> packed token streams
train/      token streams -> checkpoints
```

## corpora/

The stages run in order; each writes what the next reads.

| stage | what it does |
|---|---|
| `download/{biolip,geom,plinder}.py` | fetch a source corpus (inode-safe: archives are streamed, never expanded) |
| `build_descriptors.py` | CrossDocked tars -> sharded 33-D descriptor cache + `normalization_stats.pt` |
| `recompute_norm_stats.py` | regenerate just the stats when the schema changes, without rebuilding shards |
| `tokenize_{crossdocked,plinder,biolip,geom}.py` | descriptors -> `.bin`/`.len` token streams, one per source |
| `tokenize_affinity_{biolip,pdbbind}.py` | the same, labelled with pK for the affinity head |
| `build_docking_decoys.py`, `tokenize_decoys.py`, `concat_decoy_shards.py` | RMSD-labelled decoy poses for the rescoring head |
| `tokenize_pose_refine.py` | (corrupted pose, native pose, pocket) triples for the refiner |
| `mix.py` | concatenate token caches into one mixed-corpus cache |
| `build_hf_dataset.py` | build the CrossDocked HuggingFace dataset the descriptor stage reads |

**Why four tokenize scripts and not one `--source` flag.** Only the flush loop
was ever shared, and it now lives in `prolit.data.token_stream`. What remains is
genuinely per-source: PLINDER is zips read from memory, BioLIP is `.tar.bz2`
buckets that must be streamed in one pass to avoid O(n²) decompression, GEOM is
pickled RDKit conformers, CrossDocked is a pre-built descriptor shard cache.
Folding those into one CLI would mean four disjoint code paths behind the union
of their flags.

## train/

| script | trains |
|---|---|
| `vqvae.py` | the ProLIT tokenizer (joint, or one modality for the separate ablation) |
| `clm.py` | ProLIT-CLM, the generative decoder |
| `mlm.py` | ProLIT-MLM, the masked encoder |
| `head.py` | a pose-rescoring or affinity head on a frozen or fine-tuned MLM |
| `refiner.py` | the E(3)-equivariant flow-matching pose refiner |

Run them with the venv interpreter directly rather than `uv run`: `uv run`
re-resolves the editable install on every invocation, which is slow here and
pointless inside a job that already has the environment.

```sh
PYTHONPATH=$PWD .venv/bin/python pipelines/train/clm.py --token-dir data/lm_tokens_allatom
```
