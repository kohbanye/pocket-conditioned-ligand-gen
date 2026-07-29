"""In-house pocket-ligand VQ-VAE reconstruction (subprocess).

The own model reconstructs a protein **pocket backbone** and the **ligand** at
once, and its ``reconstruct_one`` accepts an arbitrary receptor PDB + ligand SDF
(it extracts the pocket internally). This adapter feeds it the benchmark samples
through ``scripts/own_reconstruct_cli.py``, run in the model's own uv venv, which
writes per-complex ``*_orig_pocket.pdb`` / ``*_recon.pdb`` pairs that we parse.

Because the own model defines the pocket residue subset, its ``orig_pocket`` PDBs
also serve as the shared protein input for ESM3 / FoldToken (see
:meth:`as_pocket_samples`), so all three models can be compared on the same
residues.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plbench import paths
from plbench.adapters.base import ReconstructionModel
from plbench.structio import read_backbone, read_hetatm
from plbench.types import ModalityRecon, ReconResult, Sample

_CLI = paths.REPO_ROOT / "scripts" / "own_reconstruct_cli.py"


@dataclass
class OwnRecord:
    tag: str
    orig_pocket_pdb: Path
    recon_pdb: Path


class OwnVQVAEAdapter(ReconstructionModel):
    name = "own_vqvae"
    can_protein = True
    can_ligand = True

    def __init__(
        self,
        ckpt: str | Path | None = None,
        python: str | None = None,
        out_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        **_: object,
    ) -> None:
        self.ckpt = Path(ckpt) if ckpt else paths.OWN_VQVAE_CKPT
        self.python = python or paths.OWN_MODEL_PYTHON
        self.out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_DIR / "own_recon"
        # The descriptor cache supplies the normalization stats; its schema must
        # match the checkpoint (e.g. 3dvcbp0h -> descriptor_cache_v4).
        self.cache_dir = Path(cache_dir) if cache_dir else paths.OWN_DESCRIPTOR_CACHE
        self._records: dict[str, OwnRecord] = {}
        self._results: dict[str, ReconResult] = {}

    def setup(self) -> None:
        if not self.ckpt.exists():
            raise FileNotFoundError(
                f"own VQ-VAE checkpoint missing: {self.ckpt}. "
                "Run scripts/fetch_weights.py --own."
            )

    # -- sample-set materialization --------------------------------------
    def materialize(self, samples: list[Sample]) -> list[OwnRecord]:
        """Reconstruct each (protein, ligand) sample via the own model's CLI."""
        self.setup()
        usable = [s for s in samples if s.protein_pdb and s.ligand_sdf]
        if not usable:
            raise ValueError("own model needs samples with both protein_pdb and ligand_sdf")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pairs = [
            {"id": s.sample_id, "receptor": str(s.protein_pdb), "ligand": str(s.ligand_sdf)}
            for s in usable
        ]
        pairs_path = self.out_dir / "pairs.json"
        pairs_path.write_text(json.dumps(pairs))

        cmd = [
            self.python, str(_CLI),
            "--workdir", str(paths.OWN_MODEL_WORKDIR),
            "--ckpt", str(self.ckpt.resolve()),
            "--cache-dir", str(self.cache_dir.resolve()),
            "--out-dir", str(self.out_dir.resolve()),
            "--pairs", str(pairs_path.resolve()),
        ]
        proc = subprocess.run(
            cmd, cwd=str(paths.OWN_MODEL_WORKDIR), capture_output=True, text=True
        )
        records = self._index_records()
        if not records:
            err = (proc.stderr or proc.stdout or "")[-2000:]
            raise RuntimeError(f"own recon produced no PDBs:\n{err}")
        return records

    def _index_records(self) -> list[OwnRecord]:
        self._records.clear()
        for recon in sorted(self.out_dir.glob("*_recon.pdb")):
            tag = recon.name[: -len("_recon.pdb")]
            orig_pocket = self.out_dir / f"{tag}_orig_pocket.pdb"
            if orig_pocket.exists():
                self._records[tag] = OwnRecord(tag, orig_pocket, recon)
        return list(self._records.values())

    def as_pocket_samples(self) -> list[Sample]:
        """Pocket PDBs as Samples so ESM3 / FoldToken reconstruct the same set."""
        return [
            Sample(sample_id=tag, protein_pdb=rec.orig_pocket_pdb,
                   meta={"source": "own_pocket"})
            for tag, rec in self._records.items()
        ]

    # -- reconstruction interface ----------------------------------------
    def reconstruct(self, sample: Sample) -> ReconResult:
        if sample.sample_id in self._results:
            return self._results[sample.sample_id]
        rec = self._records.get(sample.sample_id)
        if rec is None:
            return ReconResult(
                self.name, sample.sample_id, ok=False,
                error="sample not materialized; call materialize() first",
            )
        result = self._parse_record(rec)
        self._results[sample.sample_id] = result
        return result

    def _parse_record(self, rec: OwnRecord) -> ReconResult:
        modalities: list[ModalityRecon] = []

        bb_ref = read_backbone(rec.orig_pocket_pdb)
        bb_rec = read_backbone(rec.recon_pdb)
        n = min(len(bb_ref), len(bb_rec))
        ca_ref = bb_ref.ca[:n].astype(np.float64)
        ca_rec = bb_rec.ca[:n].astype(np.float64)
        if n > 0:
            modalities.append(
                ModalityRecon(
                    modality="protein_backbone",
                    ref=ca_ref, rec=ca_rec, atom_kind="CA", n_residues=int(n),
                )
            )

        _, lig_ref = read_hetatm(rec.orig_pocket_pdb)
        _, lig_rec = read_hetatm(rec.recon_pdb)
        m = min(len(lig_ref), len(lig_rec))
        lig_ref, lig_rec = lig_ref[:m].astype(np.float64), lig_rec[:m].astype(np.float64)
        if m > 0:
            modalities.append(
                ModalityRecon(
                    modality="ligand", ref=lig_ref, rec=lig_rec,
                    atom_kind="heavy", n_tokens=int(m),
                )
            )

        # Whole protein-ligand complex aligned together (pocket CA + ligand heavy
        # in one Kabsch fit). Unlike the separate rows, this also penalises drift
        # in the ligand's pose relative to the pocket.
        if n > 0 and m > 0:
            modalities.append(
                ModalityRecon(
                    modality="complex",
                    ref=np.vstack([ca_ref, lig_ref]),
                    rec=np.vstack([ca_rec, lig_rec]),
                    atom_kind="CA+heavy",
                )
            )

        return ReconResult(self.name, rec.tag, modalities=modalities)
