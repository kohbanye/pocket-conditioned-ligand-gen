"""Bio2Token reconstruction (subprocess, dedicated venv).

Bio2Token (Pillai et al., arXiv 2410.19110) is an all-atom autoencoder -- Mamba
layers with an FSQ quantizer -- released with pretrained weights for proteins,
RNA and small molecules. Two properties make it the closest published comparison
to the joint all-atom tokenizer this paper proposes:

* **One token per atom**, so the rate comparison is apples-to-apples. Its FSQ
  levels are ``[4]*6``, i.e. a 4096-entry codebook = **12 bits/atom**, the same
  rate as the capacity-matched ``separate4096`` arm.
* **It works in the input coordinate frame**, not a molecule-internal one. Its
  reconstruction therefore keeps absolute placement, which is why ``rmsd`` and
  ``kabsch_rmsd`` come out equal for it -- unlike ConfSeq or Token-Mol, whose
  SE(3)-invariant token strings drop the pose entirely.

Three modes, all fed from the all-atom NPZ dumps the own tokenizer produced, so
every model is scored on the identical pocket and ligand:

``protein``  pocket heavy atoms       (prot2token weights)
``ligand``   ligand heavy atoms       (mol2token weights)
``complex``  pocket + ligand together (bio2token weights)

**The complex mode is out of distribution.** Bio2Token was trained on proteins,
RNA and small molecules as separate structures, and its upstream PDB reader
silently discards every non-standard residue -- so no protein-ligand complex
ever reached it. The row is informative (an all-atom tokenizer *can* be handed a
complex) but must be labelled as OOD wherever it is reported.

Runs in ``.venv-bio2token`` (see ``scripts/setup_bio2token_env.sh``) because
mamba-ssm's CUDA kernels pin an exact torch build.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from plbench import paths
from plbench.adapters.base import ReconstructionModel
from plbench.types import ModalityRecon, ReconResult, Sample

_CLI = paths.REPO_ROOT / "scripts" / "bio2token_reconstruct_cli.py"

# FSQ levels [4]*6 -> 4096 codes -> 12 bits per atom.
_BITS_PER_ATOM = 12.0

_MODES = {
    "protein": ("prot2token_pretrained", "protein_allatom"),
    "ligand": ("mol2token_pretrained", "ligand"),
    "complex": ("bio2token_pretrained", "complex"),
}


class Bio2TokenAdapter(ReconstructionModel):
    """Reconstruct a pocket, a ligand, or a whole complex with Bio2Token."""

    can_protein = True
    can_ligand = True

    def __init__(
        self,
        mode: str = "complex",
        dumps: str | Path | None = None,
        python: str | None = None,
        out_dir: str | Path | None = None,
        **_: object,
    ) -> None:
        if mode not in _MODES:
            raise KeyError(f"unknown mode {mode!r}; choose from {sorted(_MODES)}")
        self.mode = mode
        self.run_id, self.modality = _MODES[mode]
        self.name = f"bio2token.{mode}"
        self.python = python or str(paths.BIO2TOKEN_PYTHON)
        # The dumps double as the shared input: same pocket, same ligand, same
        # frame as every all-atom arm, so no model is scored on its own cut.
        self.dumps = Path(dumps) if dumps else paths.OUTPUTS_DIR / "own_allatom" / "joint"
        self.out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_DIR / "bio2token" / mode
        self._dumps: dict[str, Path] = {}

    def checkpoint(self) -> Path:
        found = sorted((paths.BIO2TOKEN_CKPT_DIR / self.run_id).glob("*.ckpt"))
        if not found:
            raise FileNotFoundError(
                f"no Bio2Token checkpoint in {paths.BIO2TOKEN_CKPT_DIR / self.run_id}"
            )
        return found[0]

    def setup(self) -> None:
        self.checkpoint()
        if not Path(self.python).exists():
            raise FileNotFoundError(
                f"Bio2Token venv missing: {self.python}. "
                "Run scripts/setup_bio2token_env.sh."
            )

    # -- sample-set materialization --------------------------------------
    def materialize(self, samples: list[Sample] | None = None) -> list[str]:
        """Run the reconstruction CLI over every dump."""
        self.setup()
        wanted = {s.sample_id for s in samples} if samples else None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(  # noqa: S603
            [
                self.python, str(_CLI),
                "--repo", str(paths.BIO2TOKEN_REPO.resolve()),
                "--dumps", str(self.dumps.resolve()),
                "--out-dir", str(self.out_dir.resolve()),
                "--mode", self.mode,
                "--checkpoint", str(self.checkpoint().resolve()),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self._index_dumps()
        missing = (wanted - set(self._dumps)) if wanted else set()
        # Stale output from an earlier, smaller run must never pass for a result.
        if proc.returncode != 0 or missing:
            err = (proc.stderr or proc.stdout or "")[-2000:]
            why = []
            if proc.returncode != 0:
                why.append(f"CLI exited {proc.returncode}")
            if missing:
                why.append(f"{len(missing)} samples have no output")
            raise RuntimeError(f"bio2token.{self.mode}: {'; '.join(why)}\n{err}")
        return sorted(self._dumps)

    def _index_dumps(self) -> None:
        self._dumps = {p.stem: p for p in sorted(self.out_dir.glob("*.npz"))}

    # -- reconstruction interface ----------------------------------------
    def reconstruct(self, sample: Sample) -> ReconResult:
        if not self._dumps:
            self._index_dumps()
        dump = self._dumps.get(sample.sample_id)
        if dump is None:
            return ReconResult(
                self.name, sample.sample_id, ok=False,
                error="sample not materialized; call materialize() first",
            )
        d = np.load(dump, allow_pickle=False)
        ref, rec = d["ref"], d["rec"]
        extra = {
            "bits_protein": _BITS_PER_ATOM,
            "bits_ligand": _BITS_PER_ATOM,
            "pose_bits": 0.0,
            "arm_label": f"Bio2Token ({self.mode})",
            "arm_codebook": "FSQ 4^6 = 4096",
            "ligand_frame": "shared (input frame)",
        }
        src = None
        if self.mode in ("protein", "complex"):
            src = np.load(self.dumps / f"{sample.sample_id}.npz", allow_pickle=False)
            order = d["protein_order"]
        if self.mode == "complex":
            extra.update(
                n_protein_rows=int(d["n_protein_rows"]),
                protein_elements=[str(e) for e in src["protein_elements"][order]],
                ligand_elements=[str(e) for e in src["ligand_elements"]],
            )
        modalities = [
            ModalityRecon(
                modality=self.modality,
                ref=ref.astype(np.float64),
                rec=rec.astype(np.float64),
                atom_kind="heavy",
                n_tokens=int(d["n_tokens"]),
                extra=extra,
            )
        ]
        if src is not None:
            # A CA-only view as well. ESM3 and FoldToken reconstruct backbones and
            # nothing else, so TM-score and CA-lDDT are the only axis on which
            # they can be compared at all -- and TM-score is undefined off a
            # per-residue CA trace. Without this row Bio2Token would sit in the
            # all-atom group with no protein-tokenizer baseline beside it.
            names = [str(n).strip() for n in src["protein_atom_names"][order]]
            ca = [i for i, n in enumerate(names) if n == "CA"]
            if ca:
                modalities.append(
                    ModalityRecon(
                        modality="protein_backbone",
                        ref=ref[ca].astype(np.float64),
                        rec=rec[ca].astype(np.float64),
                        atom_kind="CA",
                        n_residues=len(ca),
                        n_tokens=int(d["n_tokens"]),
                        res_keys=[
                            (str(c), int(r))
                            for c, r in zip(
                                src["protein_chain"][order][ca],
                                src["protein_resid"][order][ca],
                                strict=True,
                            )
                        ],
                        extra=dict(extra),
                    )
                )
        return ReconResult(self.name, sample.sample_id, modalities=modalities)
