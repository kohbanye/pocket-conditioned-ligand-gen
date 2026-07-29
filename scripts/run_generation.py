"""Generate ligands for every target with the selected model(s).

Each model runs in its own interpreter (see sbddbench/paths.py) as a subprocess,
so this is normally launched once per model on a GPU node. Output layout::

    outputs/<model>/<target_id>/generated.sdf
    outputs/<model>/manifest.json      # per-target sdf path, counts, errors

Examples
--------
    # Ours (uv venv of the working copy) over all prepared targets, 100 each
    python scripts/run_generation.py --models own --n-samples 100

    # DiffSBDD on the first 10 targets
    python scripts/run_generation.py --models diffsbdd --limit 10 --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sbddbench import adapters, datasets, paths  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["own"], choices=adapters.available())
    p.add_argument("--index", type=Path, default=datasets.DEFAULT_INDEX)
    p.add_argument("--ids", nargs="*", default=None, help="restrict to these target ids")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=paths.OUTPUTS_DIR)
    args = p.parse_args()

    paths.ensure_dirs()
    targets = datasets.load_targets(args.index, limit=args.limit, ids=args.ids)
    print(f"[gen] {len(targets)} targets | models={args.models} | n_samples={args.n_samples}")

    for name in args.models:
        model = adapters.build(name)
        try:
            model.setup()
        except Exception as exc:  # noqa: BLE001
            print(f"[gen] {name}: setup failed: {exc!r}")
            continue
        manifest = []
        model_dir = args.out_dir / name
        for i, t in enumerate(targets, 1):
            out_dir = model_dir / t.target_id
            res = model._timed_generate(t, args.n_samples, out_dir)  # noqa: SLF001
            rec = asdict(res)
            rec["sdf"] = str(res.sdf) if res.sdf else None
            manifest.append(rec)
            status = "ok" if res.ok else f"FAIL: {(res.error or '')[:80]}"
            print(f"[gen] {name} {i}/{len(targets)} {t.target_id}: "
                  f"{res.n_generated}/{res.n_requested} ({res.runtime_s:.0f}s) {status}")
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        ok = sum(1 for m in manifest if m["ok"])
        print(f"[gen] {name}: {ok}/{len(manifest)} targets ok -> {model_dir}/manifest.json")


if __name__ == "__main__":
    main()
