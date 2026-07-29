"""Shared tokenizer/encoder loading + complex encoding (ported from the source repo).

Ports the ``_PoseEncoder`` recipe from ``scripts/eval_casf_rescore.py`` so pose
rescoring and affinity inference share one encode path that depends only on the
source repo's stable library layer. The fixed-pocket trick (encode protein codes
once per target, ligand codes per pose) is preserved.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ctbench.inference import ensure_source_repo_importable

ensure_source_repo_importable()

from src.config import (  # noqa: E402
    AtomVQVAETrainingConfig,
    ComplexMLMConfig,
    MLMTrainingConfig,
    PocketExtractionConfig,
    RescoreTrainingConfig,
)
from src.data.descriptors import collate_molecules  # noqa: E402
from src.data.rescore_dataset import _ligand_mask  # noqa: E402
from src.model.mlm_module import ComplexMLMModule  # noqa: E402
from src.model.rescore_module import ComplexRescoreModule  # noqa: E402
from src.model.vqvae_module import AtomVQVAEModule  # noqa: E402
from src.tokenizers.atom import (  # noqa: E402
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    precompute_receptor_atom_features_from_text,
)
from src.tokenizers.lm_vocab import AtomLMVocab  # noqa: E402
from src.tokenizers.protein import (  # noqa: E402
    _compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates_from_text,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ctbench.config import PathsConfig
    from ctbench.variants import AffinityCkpts, RescoringCkpts


def mol_to_dict(mol: Any) -> dict | None:  # noqa: ANN401
    """Convert an RDKit mol (with a conformer) to an ``atoms``/``bonds`` dict."""
    from rdkit import Chem  # noqa: PLC0415

    try:
        conf = mol.GetConformer()
    except ValueError:
        return None
    bt = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    atoms = []
    for i, a in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        atoms.append((a.GetSymbol(), p.x, p.y, p.z))
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bt.get(b.GetBondType(), 1))
        for b in mol.GetBonds()
    ]
    return {"atoms": atoms, "bonds": bonds}


def parse_mol2_multi(text: str) -> list[tuple[str, dict]]:
    """(pose_name, mol_dict) for each molecule in a multi-``@<TRIPOS>MOLECULE`` file."""
    from rdkit import Chem  # noqa: PLC0415

    out: list[tuple[str, dict]] = []
    for chunk in text.split("@<TRIPOS>MOLECULE")[1:]:
        block = "@<TRIPOS>MOLECULE" + chunk
        name = next((ln.strip() for ln in chunk.splitlines() if ln.strip()), "")
        mol = Chem.MolFromMol2Block(block, sanitize=False, removeHs=False)
        if mol is None:
            continue
        d = mol_to_dict(mol)
        if d is not None:
            out.append((name, d))
    return out


def load_vqvae(
    ckpt: Path,
    norm_stats: Path,
    codebook_size: int,
    device: torch.device,
) -> tuple[AtomVQVAEModule, np.ndarray, np.ndarray]:
    """Load the all-atom VQ-VAE tokenizer + its normalization stats (eval mode)."""
    cfg = AtomVQVAETrainingConfig()
    cfg.atom.codebook_size = codebook_size
    module = AtomVQVAEModule.load_from_checkpoint(ckpt, config=cfg, map_location=device)
    module.eval().to(device)
    norm = torch.load(norm_stats, weights_only=False)
    module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
    return module, norm["atom_mean"].numpy(), norm["atom_std"].numpy()


def load_separate_vqvae(  # noqa: PLR0913
    protein_ckpt: Path,
    protein_norm: Path,
    ligand_ckpt: Path,
    ligand_norm: Path,
    codebook_size: int,
    device: torch.device,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Load protein-only + ligand-only VQ-VAEs as one combined-code-space encoder.

    ``codebook_size`` is the PER-MODALITY sub-codebook size (e.g. 8192); the
    combined vocab (2x that) is passed separately to :func:`make_encoder`. Returns
    ``(sep, identity_mean, identity_std)`` with identity RAW-descriptor stats
    (``np.zeros(33)`` / ``np.ones(33)``): :class:`SeparateVQVAE` normalizes each
    modality internally, so :class:`ComplexEncoder` must feed it RAW descriptors.
    """
    from src.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

    sep = SeparateVQVAE.from_checkpoints(
        protein_ckpt,
        protein_norm,
        ligand_ckpt,
        ligand_norm,
        device,
        codebook_size=codebook_size,
    )
    return sep, np.zeros(33, dtype=np.float32), np.ones(33, dtype=np.float32)


def load_tokenizer(
    ckpts: RescoringCkpts | AffinityCkpts,
    paths: PathsConfig,
    device: torch.device,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Load a variant's tokenizer (joint single VQ or separate protein+ligand VQs).

    Dispatches on ``ckpts.is_separate``: the separate arm loads two single-modality
    VQ-VAEs into one combined code space (feeding identity RAW-descriptor stats);
    the joint arm loads the single combined VQ with the shared normalization stats.
    Returns ``(module, mean, std)`` ready for :func:`make_encoder`.
    """
    if ckpts.is_separate:
        pv, lv = ckpts.protein_vqvae, ckpts.ligand_vqvae
        pn, ln = ckpts.protein_norm, ckpts.ligand_norm
        if pv is None or lv is None or pn is None or ln is None:
            msg = "separate variant is missing protein/ligand vqvae or norm stats"
            raise ValueError(msg)
        # ``codebook_size`` is the COMBINED vocab (2x per-modality); each
        # single-modality sub-VQ uses half of it (16384 -> 8192, 8192 -> 4096).
        return load_separate_vqvae(
            paths.ckpt(pv),
            paths.ckpt(pn),
            paths.ckpt(lv),
            paths.ckpt(ln),
            ckpts.codebook_size // 2,
            device,
        )
    vqvae_ckpt = ckpts.vqvae
    if vqvae_ckpt is None:
        msg = "variant is missing its vqvae checkpoint"
        raise ValueError(msg)
    return load_vqvae(
        paths.ckpt(vqvae_ckpt),
        paths.norm_stats,
        ckpts.codebook_size,
        device,
    )


def load_mlm(ckpt: Path, codebook_size: int, device: torch.device) -> tuple[Any, int]:
    """Load the complex-token MLM backbone; return (model, mask_token_id)."""
    cfg = MLMTrainingConfig(model=ComplexMLMConfig(atom_codebook_size=codebook_size))
    mlm = ComplexMLMModule.load_from_checkpoint(
        ckpt,
        config=cfg,
        map_location=device,
    ).model
    mlm.eval().to(device)
    return mlm, cfg.model.mask_token_id


def load_rescorer(
    ckpt: Path,
    pooling: str,
    codebook_size: int,
    device: torch.device,
    interaction_layers: int = 0,
) -> ComplexRescoreModule:
    """Load a rescoring/affinity head (its pooling must match the checkpoint)."""
    cfg = RescoreTrainingConfig(
        model=ComplexMLMConfig(atom_codebook_size=codebook_size),
        pooling=pooling,
        head_interaction_layers=interaction_layers,
    )
    rescorer = ComplexRescoreModule.load_from_checkpoint(
        ckpt,
        config=cfg,
        map_location=device,
    )
    rescorer.eval().to(device)
    return rescorer


_RESCORE_VL_RE = re.compile(r"rescore-e\d+-vl([0-9.]+)\.ckpt$")


def _rescore_val_loss(path: Path) -> float:
    """Parse the val-loss encoded in a ``rescore-eNN-vlX.XXXX.ckpt`` filename."""
    m = _RESCORE_VL_RE.search(path.name)
    return float(m.group(1)) if m is not None else float("inf")


def resolve_rescore_ckpt(source_repo: Path, spec: str) -> Path:
    """Resolve a head checkpoint: an exact ``*.ckpt`` path or a run-name to glob.

    If ``spec`` ends in ``.ckpt`` it is treated as a path relative to the source
    repo. Otherwise it is a rescore run-name whose ``checkpoints`` directory is
    globbed for ``rescore-*.ckpt``, returning the one with the LOWEST val-loss.
    """
    if spec.endswith(".ckpt"):
        return source_repo / spec
    ckpt_dir = source_repo / "pocket-ligand-rescore" / spec / "checkpoints"
    candidates = sorted(ckpt_dir.glob("rescore-*.ckpt"))
    if not candidates:
        msg = f"no rescore checkpoints found for head {spec!r} under {ckpt_dir}"
        raise FileNotFoundError(msg)
    return min(candidates, key=_rescore_val_loss)


def ligand_mask(seq: Sequence[int]) -> np.ndarray:
    """0/1 mask marking ligand-token positions in an assembled sequence."""
    return _ligand_mask(np.asarray(seq))


class ComplexEncoder:
    """Fixed-pocket encoder: protein codes once per target, ligand codes per pose."""

    def __init__(  # noqa: PLR0913
        self,
        module: AtomVQVAEModule,
        mean: np.ndarray,
        std: np.ndarray,
        vocab: AtomLMVocab,
        device: torch.device,
        pocket_cfg: PocketExtractionConfig,
    ) -> None:
        self.module = module
        self.mean = mean
        self.std = std
        self.vocab = vocab
        self.device = device
        self.pocket_cfg = pocket_cfg
        self.prot_desc = ProteinAtomDescriptor()
        self.lig_desc = LigandAtomDescriptor()

    def _encode(self, desc: np.ndarray) -> list[int]:
        x, mask = collate_molecules(
            [torch.from_numpy((desc - self.mean) / self.std).float()],
        )
        idx = self.module.vqvae.encode_batch(
            x.to(self.device),
            mask.to(self.device),
        ).cpu()
        return idx[0][mask[0]].tolist()

    def setup_pocket(
        self,
        protein_text: str,
        native_heavy: np.ndarray,
    ) -> tuple[list[int], Any] | None:
        """Extract the pocket around a reference ligand -> (protein_codes, frame)."""
        precomp = precompute_pocket_atom_candidates_from_text(protein_text)
        pocket = extract_pocket_atoms_from_candidates(
            precomp,
            native_heavy,
            self.pocket_cfg,
        )
        if pocket is None or pocket.atom_coords.shape[0] == 0:
            return None
        feats = precompute_receptor_atom_features_from_text(protein_text)
        frame = _compute_canonical_frame(pocket.ca_coords.astype(np.float64))
        prot_desc, _ = self.prot_desc.compute(pocket, feats, frame)
        if prot_desc.shape[0] == 0:
            return None
        return self._encode(prot_desc), frame

    def ligand_seq(
        self,
        p_codes: list[int],
        mol: dict,
        frame: Any,  # noqa: ANN401  (source-repo pocket frame)
    ) -> list[int] | None:
        """Assemble the ``<p>pocket</p><l>ligand</l>`` token sequence for one pose."""
        lig_desc, _e, _m = self.lig_desc.compute(mol["atoms"], mol["bonds"], frame)
        if lig_desc.shape[0] == 0:
            return None
        return self.vocab.build_sequence(p_codes, self._encode(lig_desc))

    def ligand_seqs_batch(
        self,
        p_codes: list[int],
        mols: list[dict],
        frame: Any,  # noqa: ANN401  (source-repo pocket frame)
    ) -> list[list[int] | None]:
        """Encode many ligand poses in one VQ call (decoy-scoring fast path)."""
        descs = [self.lig_desc.compute(m["atoms"], m["bonds"], frame)[0] for m in mols]
        valid = [(i, d) for i, d in enumerate(descs) if d.shape[0] > 0]
        out: list[list[int] | None] = [None] * len(mols)
        if not valid:
            return out
        tensors = [
            torch.from_numpy((d - self.mean) / self.std).float() for _, d in valid
        ]
        x, mask = collate_molecules(tensors)
        idx = self.module.vqvae.encode_batch(
            x.to(self.device),
            mask.to(self.device),
        ).cpu()
        for k, (i, _) in enumerate(valid):
            out[i] = self.vocab.build_sequence(p_codes, idx[k][mask[k]].tolist())
        return out


def make_encoder(  # noqa: PLR0913
    module: AtomVQVAEModule,
    mean: np.ndarray,
    std: np.ndarray,
    codebook_size: int,
    device: torch.device,
    max_residues: int,
) -> ComplexEncoder:
    """Construct a :class:`ComplexEncoder` with the standard vocab + pocket config."""
    return ComplexEncoder(
        module,
        mean,
        std,
        AtomLMVocab(codebook_size=codebook_size),
        device,
        PocketExtractionConfig(max_residues=max_residues),
    )
