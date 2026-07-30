# scripts/

The ProLIT entry points a benchmark drives as a subprocess. Nothing else.

| script | driven by |
|---|---|
| `generate_ligands_for_target.py` | sbdd-bench's `own` adapter; pose-rescoring-bench |
| `generate_ligands_3d.py` | pose-rescoring-bench (CrossDocked test pockets) |
| `eval_casf_rescore.py` | pose-rescoring-bench (docking power) |
| `eval_casf_scoring.py` | pose-rescoring-bench (scoring / ranking power) |
| `prepare_target.py`, `dock_vina.py` | the generators above |

The benchmarks run these rather than importing them, because generation needs
the model's environment while scoring is done in the benchmark's. That
subprocess boundary is the only reason this directory exists.

## Before adding a file here

Ask which of these it is:

- **reusable logic** — put it in `src/prolit/` and import it. This is where
  `PoseEncoder`, `parse_mol2_multi`, `obabel_mol` and the Vina helpers ended up
  after each was found living here and being reached for from two other layers.
- **corpus construction or training** — `pipelines/`.
- **benchmark-specific** — that benchmark's own `scripts/`. The two CASF
  baseline generators moved to `benchmarks/pose-rescoring-bench/scripts/` for
  exactly this reason: they score Vina and Boltz-2, not ProLIT.

`tests/test_layering.py` caps this directory at 12 entry points, so growing it
is a decision rather than a habit. It held 62 once.
