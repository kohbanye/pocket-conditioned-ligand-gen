"""Target-set loader.

A "target set" is a directory of prepared pockets plus an ``index.json`` listing
them, produced by ``scripts/prepare_targets.py``. The canonical set is the
**CrossDocked2020 test split** (the 100 pockets DiffSBDD / TargetDiff / DiffGui
all report on); ``prepare_targets.py`` can also build a single named target
(e.g. EGFR 2ITY) for quick checks. The loader is agnostic to which — it just
reads the index.
"""

from __future__ import annotations

import json
from pathlib import Path

from sbddbench import paths
from sbddbench.types import Target

DEFAULT_INDEX = paths.TARGETS_DIR / "index.json"


def load_targets(
    index_path: str | Path = DEFAULT_INDEX,
    limit: int | None = None,
    ids: list[str] | None = None,
) -> list[Target]:
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"target index missing: {index_path}. Run scripts/prepare_targets.py first."
        )
    records = json.loads(index_path.read_text())
    if ids:
        wanted = set(ids)
        records = [r for r in records if r["target_id"] in wanted]
    if limit is not None:
        records = records[:limit]

    base = index_path.parent
    out: list[Target] = []
    for r in records:
        # ``record=r`` binds the current row: the closure is only used inside
        # this iteration, and binding keeps it that way if it ever escapes.
        def _p(key, record=r):
            v = record.get(key)
            if v is None:
                return None
            p = Path(v)
            return p if p.is_absolute() else (base / p)

        box = r.get("box")
        if box is None and r.get("box_json"):
            box = json.loads((_p("box_json")).read_text())
        out.append(
            Target(
                target_id=r["target_id"],
                receptor_pdb=_p("receptor_pdb"),
                ref_ligand_sdf=_p("ref_ligand_sdf"),
                pocket_pdb=_p("pocket_pdb"),
                receptor_pdbqt=_p("receptor_pdbqt"),
                box=box,
                meta=r.get("meta", {}),
            )
        )
    return out
