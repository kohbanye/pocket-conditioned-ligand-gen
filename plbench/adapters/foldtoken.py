"""FoldToken4 reconstruction via the upstream ``reconstruct.py`` (subprocess).

FoldToken's deps (chroma, older torch) collide with the bench env, and its
script is CUDA-only, so we shell out to a dedicated interpreter set by
``PLBENCH_FOLDTOKEN_PYTHON``. The script reads a folder of PDBs and writes
``<title>_pred.pdb`` plus ``vqids.json`` into ``<out>_level{level}/``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from plbench import paths
from plbench.adapters.base import ReconstructionModel
from plbench.structio import read_backbone
from plbench.types import ModalityRecon, ReconResult, Sample


class FoldTokenAdapter(ReconstructionModel):
    name = "foldtoken"
    can_protein = True
    can_ligand = False

    def __init__(self, level: int = 8, python: str | None = None, **_: object) -> None:
        self.level = level
        self.python = python or paths.FOLDTOKEN_PYTHON

    def setup(self) -> None:
        if not paths.FOLDTOKEN_CKPT.exists():
            raise FileNotFoundError(
                f"FoldToken checkpoint missing: {paths.FOLDTOKEN_CKPT}. "
                "Run scripts/fetch_weights.py --foldtoken."
            )

    def reconstruct(self, sample: Sample) -> ReconResult:
        return self.reconstruct_batch([sample])[0]

    def reconstruct_batch(self, samples) -> list[ReconResult]:
        samples = [s for s in samples if s.protein_pdb is not None]
        if not samples:
            return []
        self.setup()
        work = Path(tempfile.mkdtemp(prefix="plbench_foldtoken_"))
        in_dir = work / "in"
        out_base = work / "out"
        in_dir.mkdir()
        for s in samples:
            shutil.copy(s.protein_pdb, in_dir / f"{s.sample_id}.pdb")

        cmd = [
            self.python,
            "foldtoken/reconstruct.py",
            "--path_in", str(in_dir),
            "--path_out", str(out_base),
            "--config", str(paths.FOLDTOKEN_CONFIG),
            "--checkpoint", str(paths.FOLDTOKEN_CKPT),
            "--level", str(self.level),
        ]
        env = {"PYTHONPATH": str(paths.FOLDTOKEN_REPO)}
        proc = subprocess.run(
            cmd, cwd=str(paths.FOLDTOKEN_REPO), capture_output=True, text=True,
            env={**_os_environ(), **env},
        )
        out_dir = Path(f"{out_base}_level{self.level}")
        if not out_dir.exists():
            err = (proc.stderr or proc.stdout or "")[-2000:]
            return [
                ReconResult(self.name, s.sample_id, ok=False, error=f"foldtoken failed: {err}")
                for s in samples
            ]

        vqids = {}
        vq_path = out_dir / "vqids.json"
        if vq_path.exists():
            vqids = json.loads(vq_path.read_text())

        results = []
        for s in samples:
            results.append(self._read_one(s, in_dir, out_dir, vqids))
        shutil.rmtree(work, ignore_errors=True)
        return results

    def _read_one(self, sample: Sample, in_dir, out_dir, vqids) -> ReconResult:
        pred = out_dir / f"{sample.sample_id}_pred.pdb"
        if not pred.exists():
            return ReconResult(
                self.name, sample.sample_id, ok=False, error="no pred pdb written"
            )
        ref = read_backbone(in_dir / f"{sample.sample_id}.pdb")
        rec = read_backbone(pred)
        n = min(len(ref), len(rec))
        res_keys = [
            (str(c), int(r))
            for c, r in zip(ref.chain_ids[:n], ref.res_ids[:n], strict=False)
        ]
        modality = ModalityRecon(
            modality="protein_backbone",
            ref=ref.ca[:n].astype(np.float64),
            rec=rec.ca[:n].astype(np.float64),
            atom_kind="CA",
            n_residues=int(n),
            n_tokens=len(vqids.get(sample.sample_id, [])) or None,
            res_keys=res_keys,
        )
        return ReconResult(self.name, sample.sample_id, modalities=[modality])


def _os_environ() -> dict:
    import os

    return dict(os.environ)
