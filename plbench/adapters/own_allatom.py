"""In-house **all-atom** pocket-ligand tokenizer reconstruction (subprocess).

Successor to :mod:`plbench.adapters.own_vqvae`, which drives the older
residue-level tokenizer. The all-atom family encodes pocket atoms and ligand
atoms with one shared 33-D descriptor, so a single codebook can cover both, and
the ablation question is what that sharing costs and buys.

One adapter instance = one **arm**. The arms in :data:`ARMS` span the two design
axes the paper argues about:

* **codebook** — one shared book (``joint``) vs a hard partition into a
  protein-only and a ligand-only book (``separate*``). No partition can match a
  shared book on codebook vectors *and* on bits/atom at once, so the separate
  arms come in a capacity-matched and a rate-matched variant.
* **frame** — the ligand encoded in the shared pocket frame (placement is in
  every atom token, ``pose_bits=0``) vs in its own canonical frame like a
  single-modality ligand tokenizer (tokens are SE(3)-invariant, so the pose must
  be transmitted separately and ``pose_bits`` prices it).

``binning`` needs no weights at all: it discretizes space on a grid at a
comparable rate and shows what the learned codebook is worth.

Reconstruction runs through ``scripts/own_allatom_reconstruct_cli.py`` in the
source repo's own venv, which dumps one NPZ per complex (per-atom
correspondence, ligand bonds, token counts) that this adapter turns into
:class:`~plbench.types.ModalityRecon` rows.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from plbench import paths
from plbench.adapters.base import ReconstructionModel
from plbench.types import ModalityRecon, ReconResult, Sample

_CLI = paths.REPO_ROOT / "scripts" / "own_allatom_reconstruct_cli.py"
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


@dataclass
class Arm:
    """One tokenizer configuration to evaluate."""

    name: str
    label: str
    protein_run: str | None = None
    ligand_run: str | None = None
    protein_norm: Path | None = None
    ligand_norm: Path | None = None
    ligand_frame: str = "pocket"
    pose_bits: int | None = None
    kind: str = "vq"
    codebook: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _cache(name: str) -> Path:
    return paths.OWN_ALLATOM_CACHE / name


ARMS: dict[str, Arm] = {
    "joint": Arm(
        name="joint",
        label="Joint (one shared codebook)",
        protein_run="xzkjxu9q",
        ligand_run="xzkjxu9q",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
    ),
    "separate": Arm(
        name="separate",
        label="Separate 8192+8192 (rate-matched)",
        protein_run="protein-vqvae",
        ligand_run="ligand-vqvae",
        protein_norm=_cache("normalization_stats_protein.pt"),
        ligand_norm=_cache("normalization_stats_ligand.pt"),
        codebook="2 books (8192+8192)",
        notes="same bits/atom as joint, 2x the codebook vectors",
    ),
    "separate4096": Arm(
        name="separate4096",
        label="Separate 4096+4096 (capacity-matched)",
        protein_run="protein-vqvae-4096",
        ligand_run="ligand-vqvae-4096",
        protein_norm=_cache("normalization_stats_protein.pt"),
        ligand_norm=_cache("normalization_stats_ligand.pt"),
        codebook="2 books (4096+4096)",
        notes="same codebook vectors and vocab as joint, 12 bits/atom",
    ),
    "binning": Arm(
        name="binning",
        label="Coordinate binning (no training)",
        kind="binning",
        codebook="grid (10^3 cells x 12 elements)",
        notes="no learned parameters; the rate-matched floor",
    ),
}

# Ligand-own-frame arms sweep the pose budget instead of fixing one, so the paper
# can report the break-even: how many extra tokens a single-modality ligand
# tokenizer must spend on the rigid transform before its interface metrics match
# a shared-frame tokenizer, which spends none. Fixing a single budget would be
# arbitrary and invites "why not one more token?".
_POSE_SWEEP = [(None, "oracle"), (39, "3tok"), (26, "2tok"), (20, "1.5tok"), (13, "1tok")]
for _bits, _tag in _POSE_SWEEP:
    ARMS[f"localframe_{_tag}"] = Arm(
        name=f"localframe_{_tag}",
        label=f"Ligand-own-frame + {_tag} pose",
        protein_run="protein-vqvae",
        ligand_run="ligand-vqvae-localframe",
        protein_norm=_cache("normalization_stats_protein.pt"),
        ligand_norm=paths.OWN_LOCALFRAME_CACHE / "normalization_stats_ligand.pt",
        ligand_frame="local",
        pose_bits=_bits,
        codebook="2 books (8192+8192)",
        notes="SE(3)-invariant ligand tokens; pose transmitted separately",
    )


def best_checkpoint(run: str, min_epoch: int = 90) -> Path | None:
    """Lowest-``val/atom_coord`` checkpoint of a *finished* run, or None.

    The '/' in the monitored metric name makes every checkpoint its own
    directory, so files land at
    ``<run>/checkpoints/atomvqvae-epoch=NN-val/atom_coord=X.ckpt``.

    ``min_epoch`` refuses checkpoints from a run that is still training. Without
    it a half-trained VQ silently becomes a row in the paper's ablation table.
    """
    ckpt_dir = paths.OWN_VQ_RUNS_DIR / run / "checkpoints"
    found = []
    for path in ckpt_dir.glob("*/atom_coord=*.ckpt"):
        epoch = int(path.parent.name.split("epoch=")[1].split("-")[0])
        if epoch >= min_epoch:
            found.append((float(path.stem.split("=")[-1]), epoch, path))
    return min(found)[2] if found else None


class OwnAllAtomAdapter(ReconstructionModel):
    """Reconstruct pocket + ligand with one all-atom tokenizer arm."""

    can_protein = True
    can_ligand = True

    def __init__(
        self,
        arm: str = "joint",
        python: str | None = None,
        out_dir: str | Path | None = None,
        min_epoch: int = 90,
        device: str | None = None,
        **_: object,
    ) -> None:
        if arm not in ARMS:
            raise KeyError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
        self.arm = ARMS[arm]
        self.name = f"own_allatom.{arm}"
        self.python = python or paths.OWN_MODEL_PYTHON
        self.out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_DIR / "own_allatom" / arm
        self.min_epoch = min_epoch
        # None lets the CLI pick; "cpu" is the escape hatch when the GPU is busy
        # (reconstruction is small, so CPU is slow but perfectly workable).
        self.device = device
        self._dumps: dict[str, Path] = {}

    # -- setup -----------------------------------------------------------
    def arm_spec(self) -> dict:
        """Resolve the arm to concrete checkpoint paths for the CLI."""
        spec: dict = {
            "kind": self.arm.kind,
            "ligand_frame": self.arm.ligand_frame,
            "pose_bits": self.arm.pose_bits,
        }
        if self.arm.kind == "binning":
            return spec
        for side in ("protein", "ligand"):
            run = getattr(self.arm, f"{side}_run")
            ckpt = best_checkpoint(run, self.min_epoch)
            if ckpt is None:
                raise FileNotFoundError(
                    f"arm {self.arm.name!r}: no {side} checkpoint past epoch "
                    f"{self.min_epoch} in {paths.OWN_VQ_RUNS_DIR / run}. "
                    "Still training, or the run name is wrong."
                )
            norm = getattr(self.arm, f"{side}_norm")
            if not norm.exists():
                raise FileNotFoundError(f"arm {self.arm.name!r}: missing {norm}")
            spec[f"{side}_ckpt"] = str(ckpt)
            spec[f"{side}_norm"] = str(norm)
        return spec

    def setup(self) -> None:
        self.arm_spec()

    @classmethod
    def ready_arms(cls, min_epoch: int = 90) -> list[str]:
        """Arms whose weights exist and are past ``min_epoch`` (binning always)."""
        ready = []
        for name in ARMS:
            try:
                cls(arm=name, min_epoch=min_epoch).arm_spec()
            except FileNotFoundError:
                continue
            ready.append(name)
        return ready

    # -- sample-set materialization --------------------------------------
    def materialize(self, samples: list[Sample]) -> list[str]:
        """Run the source repo's CLI over every (protein, ligand) sample."""
        spec = self.arm_spec()
        usable = [s for s in samples if s.protein_pdb and s.ligand_sdf]
        if not usable:
            raise ValueError("all-atom arms need samples with both protein_pdb and ligand_sdf")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "arm.json").write_text(json.dumps(spec, indent=2))
        (self.out_dir / "pairs.json").write_text(
            json.dumps(
                [
                    {
                        "id": s.sample_id,
                        "receptor": str(s.protein_pdb),
                        "ligand": str(s.ligand_sdf),
                    }
                    for s in usable
                ]
            )
        )
        cmd = [
            self.python, str(_CLI),
            "--workdir", str(paths.OWN_MODEL_WORKDIR),
            "--arm", str((self.out_dir / "arm.json").resolve()),
            "--pairs", str((self.out_dir / "pairs.json").resolve()),
            "--out-dir", str(self.out_dir.resolve()),
        ]
        if self.device:
            cmd += ["--device", self.device]
        cmd += ["--receptor-cache", str(paths.RECEPTOR_CACHE.resolve())]
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(paths.OWN_MODEL_WORKDIR),
            capture_output=True,
            text=True,
            check=False,
        )
        self._index_dumps()
        # Dumps persist between runs, so "some NPZ exists" proves nothing: a run
        # that died on the first complex would otherwise be silently scored on
        # whatever a previous, smaller run left behind, and the summary table
        # would look complete. Demand the exit code AND full coverage.
        wanted = {s.sample_id for s in usable}
        missing = wanted - set(self._dumps)
        if proc.returncode != 0 or missing:
            err = (proc.stderr or proc.stdout or "")[-2000:]
            why = []
            if proc.returncode != 0:
                why.append(f"CLI exited {proc.returncode}")
            if missing:
                why.append(f"{len(missing)}/{len(wanted)} complexes have no dump")
            raise RuntimeError(
                f"arm {self.arm.name!r}: reconstruction failed ({'; '.join(why)}). "
                f"Any dumps already in {self.out_dir} are from an earlier run and "
                f"are NOT a valid result for this one.\n{err}"
            )
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
        return self._parse_dump(sample.sample_id, dump)

    def _parse_dump(self, sample_id: str, dump: Path) -> ReconResult:
        d = np.load(dump, allow_pickle=False)
        prot_ref, prot_rec = d["protein_ref"], d["protein_rec"]
        lig_ref, lig_rec = d["ligand_ref"], d["ligand_rec"]
        prot_elements = [str(e) for e in d["protein_elements"]]
        lig_elements = [str(e) for e in d["ligand_elements"]]
        atom_names = [str(n) for n in d["protein_atom_names"]]
        res_keys_all = [
            (str(c), int(r)) for c, r in zip(d["protein_chain"], d["protein_resid"], strict=True)
        ]
        rate = {
            "bits_protein": float(d["bits_protein"]),
            "bits_ligand": float(d["bits_ligand"]),
            "pose_bits": float(d["pose_bits"]),
            "arm_label": self.arm.label,
            "arm_codebook": self.arm.codebook,
            "ligand_frame": self.arm.ligand_frame,
        }

        modalities: list[ModalityRecon] = []
        # CA-only view so this arm lines up with ESM3 / FoldToken, which
        # reconstruct backbones and nothing else.
        ca = [i for i, n in enumerate(atom_names) if n.strip() == "CA"]
        if ca:
            modalities.append(
                ModalityRecon(
                    modality="protein_backbone",
                    ref=prot_ref[ca], rec=prot_rec[ca], atom_kind="CA",
                    n_residues=len(ca),
                    n_tokens=int(d["n_tokens_protein"]),
                    res_keys=[res_keys_all[i] for i in ca],
                    extra=dict(rate),
                )
            )
        # All-atom view: what this tokenizer actually reconstructs, and the only
        # scope on which side-chain geometry at the interface is visible.
        bb = [i for i, n in enumerate(atom_names) if n.strip() in _BACKBONE_ATOMS]
        modalities.append(
            ModalityRecon(
                modality="protein_allatom",
                ref=prot_ref, rec=prot_rec, atom_kind="heavy",
                n_residues=len({k for k in res_keys_all}),
                n_tokens=int(d["n_tokens_protein"]),
                res_keys=res_keys_all,
                extra={**rate, "n_backbone_atoms": len(bb)},
            )
        )
        modalities.append(
            ModalityRecon(
                modality="ligand",
                ref=lig_ref, rec=lig_rec, atom_kind="heavy",
                n_tokens=int(d["n_tokens_ligand"]),
                extra={
                    **rate,
                    "elements": lig_elements,
                    "bonds": [(int(a), int(b)) for a, b, _ in d["ligand_bonds"]],
                    "bond_orders": [int(o) for *_, o in d["ligand_bonds"]],
                },
            )
        )
        # Protein and ligand stacked in the frame they were reconstructed in --
        # no per-modality superposition, so this is where a lost binding pose
        # actually shows up.
        modalities.append(
            ModalityRecon(
                modality="complex",
                ref=np.vstack([prot_ref, lig_ref]),
                rec=np.vstack([prot_rec, lig_rec]),
                atom_kind="heavy",
                n_tokens=int(d["n_tokens_protein"]) + int(d["n_tokens_ligand"]),
                extra={
                    **rate,
                    "n_protein_rows": int(prot_ref.shape[0]),
                    "protein_elements": prot_elements,
                    "ligand_elements": lig_elements,
                },
            )
        )
        return ReconResult(self.name, sample_id, modalities=modalities)
