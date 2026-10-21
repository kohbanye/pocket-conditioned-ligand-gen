# GIGN, measured in this environment, is well ahead of us

2026-09-14. Published numbers were not reused: GIGN's released code and its
three released checkpoints were run here on the same 285 CASF-2016 core
complexes our arms are scored on, and the resulting per-complex dumps carry the
same schema, the same labels and the same cluster ids.

| GIGN checkpoint | R | rho |
|---|---:|---:|
| repeat0 | 0.8448 | 0.6860 |
| repeat1 | 0.8410 | 0.7316 |
| repeat2 | 0.8323 | 0.7105 |
| **mean of all three** (canonical `results/gign/scoring.csv`) | **0.8550** | **0.7702** |
| z-sum of all three | 0.8550 | 0.7667 |

The two combinations differ only in how the three are pooled: the mean keeps
pK units, the z-sum standardises each member first. R is identical to four
decimals; rho differs by 0.0035, because standardising reweights members
whose spreads differ. The bench stores the mean.

All three are scored and reported. Picking the best repeat by its CASF result
would be selection on the test set -- the rule our own arms are held to -- and
the z-sum is the same treatment our seed families get, so the last row is the
like-for-like comparison.

## The table at matched 285-complex coverage

| | R | rho |
|---|---:|---:|
| **GIGN, mean of 3** (measured) | **0.8550** | **0.7702** |
| ours `e250_union_noconf` ens(5) | 0.8140 | 0.6895 |
| GenScore (measured) | 0.8127 | 0.7281 |
| ours all-families ens(44), zero selection | 0.8108 | 0.7175 |
| ours `e250_union_clean` ens(3) | 0.8075 | 0.7105 |
| ours `e250_rot1` ens(6) | 0.7913 | 0.6947 |

GIGN beats our best by **+0.041 R and +0.049 rho**. It also beats GenScore on
both, which is consistent with it being the newer method.

## Reproduction notes

Three of my own bugs cost a cycle each and are worth recording so the harness is
trusted: the pocket cutoff was passed as `5.0` to the dataset and `5` to the
preprocessing, so the dataset looked for `_5.0A.rdkit` while preprocessing had
written `_5A.rdkit`; `zip(..., strict=True)` is Python 3.10+ and the env is 3.9;
and PyG 2.0.3 hard-imports `torch_sparse`, which must be version-matched to
torch 1.10.2 or pip compiles a current one that calls `torch.sparse_csc_tensor`
and fails at import.

Environment: `baselines/envs/gign`, CPU only (285 complexes needs no GPU, and it
avoids the CUDA matrix entirely). torch 1.10.2 / PyG 2.0.3 / torch_sparse 0.6.13
/ torch_scatter 2.0.9. MKL had to be pinned below 2024 -- torch 1.10 expects
`iJIT_NotifyEvent`, which newer MKL removed.

Inputs are a staged tree of real directories holding symlinks to the CASF files,
because GIGN's preprocessing writes its pocket, `.rdkit` and `.pyg` artifacts
next to its inputs and `data/casf2016` has to stay pristine. Seven ligands
produced OpenBabel kekulization warnings; all 285 still yielded graphs.

## What this does not settle

GIGN uses **explicit interaction edges** (every ligand-pocket atom pair within
5 A becomes a graph edge), so on the framing that our peer group is methods
without hand-specified contacts, it is not the like-for-like comparison -- it is
evidence about what explicit interaction modelling buys.

It also trains on PDBbind with the core set removed by id, and the published
literature on this benchmark reports that CASF numbers for PDBbind-trained
models are inflated by train/test similarity. Measuring GIGN here confirms the
number is real in this environment; it does not make 0.8550 and our 0.8108
like-for-like, because the training corpora and their exclusion regimes differ.

The peer group without explicit contacts -- KDEEP (R 0.82) and DeepAtom
(R 0.831) -- **cannot be measured here**: KDEEP is served only through
PlayMolecule and DeepAtom's paper does not publish weights. Those two rows can
only ever be published values, and both are single models while our 0.8140 is a
five-member ensemble.
