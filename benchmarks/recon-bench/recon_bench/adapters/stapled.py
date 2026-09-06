"""The baseline a reviewer will ask for: two existing tokenizers stapled together.

ProLIT's claim is that a pocket and its ligand should be encoded in **one**
vocabulary and **one** frame. The obvious objection is that you could just
concatenate the best published protein-only tokenizer with the best published
ligand-only one and get the same thing for free. This adapter builds exactly
that and measures it.

Each half is the strongest available in its modality on this very benchmark
(``docs/results/2026-09-05_posebusters.md``), so this is not a straw man:

* **ESM3 structure tokenizer** for the pocket -- 1.202 A pocket-scope Kabsch,
  0.907 lDDT, better than FoldToken4 and Bio2Token.
* **ConfSeq** for the ligand -- 0.699 PoseBusters validity and 0.040 A bond
  MAE against ProLIT's 0.491 and 0.122. ConfSeq *beats* ProLIT on ligand
  chemistry, and it is rule-based, so there is no training contamination to
  argue about either.

What the concatenation cannot do is say how the two stand relative to each
other. ESM3 structure tokens describe per-residue local geometry; ConfSeq is
SMILES plus discretized dihedrals and bond angles, explicitly SE(3)-invariant.
Neither carries the ligand's placement in the receptor, so the stapled stream is
missing the rigid transform entirely -- six degrees of freedom about which it
says nothing. Scoring interface metrics on it without addressing that would
measure whatever placement *we* chose, not the representation.

So the baseline is handed the missing information as an explicit budget:
``pose_bits`` bits of quantized rigid transform, priced by
:mod:`prolit.tokenizers.pose_budget` -- the same quantizer the ``localframe_*``
arms use, so every ligand-own-frame row in the table sits on one curve.
``pose_bits=None`` is oracle placement, the unattainable ceiling. ProLIT spends
**zero** bits here, which is the entire point: its atoms are spherical
coordinates in a shared pocket frame, so placement is already in the tokens.

Read the resulting rows as a rate argument, never as a bare win: "ProLIT reaches
this interface quality at 13 bits/atom and no placement channel; the stapled
construction needs N additional bits to come level."

**Scope.** ESM3 reconstructs backbone only (N, CA, C), so the complex rows here
are backbone-scope. ProLIT's published complex rows are all-heavy-atom and must
not be put in the same column -- score ProLIT at backbone scope for the
comparison, or state both scopes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from recon_bench import paths
from recon_bench.adapters.base import ReconstructionModel
from recon_bench.adapters.esm3 import chain_break_layout
from recon_bench.structio import Backbone, read_backbone
from recon_bench.types import ModalityRecon, ReconResult, Sample

#: ESM3's structure codebook. 4096 entries -> 12 bits per residue token.
ESM3_CODEBOOK_BITS = 12.0

#: ConfSeq's vocabulary, measured over this benchmark's own 428 ligands
#: (all of which encode): 391 distinct tokens -> log2 = 8.61 bits. Charged
#: uniformly, the way ProLIT is charged log2(8192) = 13 rather than the
#: entropy of its code usage, so the two rates are the same kind of number.
#: For reference the unigram entropy is 5.44 bits/token, and ConfSeq spends
#: 2.88 tokens per heavy atom against ProLIT's 1.00.
CONFSEQ_VOCAB = 391
CONFSEQ_BITS = float(np.log2(CONFSEQ_VOCAB))

#: A residue joins the pocket when its CA is within this of any ligand heavy
#: atom. Stated rather than inherited so the same rule can be applied to
#: whatever ProLIT row this is compared against.
POCKET_RADIUS = 10.0


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation taking ``mobile`` onto ``target``."""
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, tc - rot @ mc


class StapledAdapter(ReconstructionModel):
    """ESM3 pocket tokens + ConfSeq ligand tokens + a priced placement channel."""

    can_protein = True
    can_ligand = True

    def __init__(
        self,
        pose_bits: int | None = 39,
        *,
        pocket_radius: float = POCKET_RADIUS,
        protein_scope: str = "full",
        seed: int = 0,
        device: str | None = None,
        **_: object,
    ) -> None:
        self.pose_bits = pose_bits
        self.name = (
            "stapled_oracle" if pose_bits is None else f"stapled_pose{pose_bits}"
        )
        self.pocket_radius = pocket_radius
        # "full" (default): ESM3 encodes the whole chain, its native task, and
        # the pocket residues are read out of that. "pocket" encodes only the
        # pocket residues -- which sounds like the tighter comparison and is
        # actually a handicap we would be imposing: a pocket is discontiguous in
        # the chain, ESM3 has no way to be told that beyond a chain break, and
        # renumbering it 1..L presents consecutive residues that sit angstroms
        # apart as neighbours. Measured on three PoseBusters complexes that
        # costs 8.48 A of backbone Kabsch against the 1.202 A ESM3 reaches on
        # the same residues when it sees the chain. The baseline gets the
        # setting it is good at.
        self.protein_scope = protein_scope
        self.seed = seed
        self.device = device
        self._esm = None
        self._confseq = None

    # -- setup ------------------------------------------------------------
    def setup(self) -> None:
        if self._esm is not None:
            return
        import torch
        from esm.pretrained import (
            ESM3_structure_decoder_v0,
            ESM3_structure_encoder_v0,
        )
        from esm.utils.constants import esm3 as C

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._esm = {
            "encoder": ESM3_structure_encoder_v0(self.device),
            "decoder": ESM3_structure_decoder_v0(self.device),
            "special": C.VQVAE_SPECIAL_TOKENS,
        }
        demo = paths.CONFSEQ_REPO / "demo"
        if str(demo) not in sys.path:
            sys.path.insert(0, str(demo))
        # The pose quantizer lives in prolit so that this benchmark and the LM
        # corpus builder price the same transform identically. prolit is not a
        # dependency of this env -- ESM3 pins a fork of transformers and must
        # not share one -- but the module it is needed from is numpy-only, so
        # the source tree goes on the path rather than into the lockfile.
        src = paths.OWN_MODEL_WORKDIR / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import ConfSeq  # type: ignore[import-not-found]
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        self._confseq = ConfSeq

    # -- the two halves ---------------------------------------------------
    def _esm3_round_trip(
        self, coords: np.ndarray, residue_index: np.ndarray, breaks: np.ndarray
    ) -> np.ndarray:
        """Encode and decode one ESM3 input layout; returns (L, 3, 3) N/CA/C."""
        import torch

        enc, dec = self._esm["encoder"], self._esm["decoder"]
        special = self._esm["special"]
        x = torch.from_numpy(coords).float().unsqueeze(0).to(self.device)
        ridx = torch.from_numpy(residue_index).long().unsqueeze(0).to(self.device)
        brk = torch.from_numpy(breaks).to(self.device)
        with torch.no_grad():
            _, tokens = enc.encode(x, residue_index=ridx)
            if brk.numel():
                tokens[:, brk] = special["CHAINBREAK"]
            tokens = torch.nn.functional.pad(tokens, (1, 1), value=0)
            tokens[:, 0] = special["BOS"]
            tokens[:, -1] = special["EOS"]
            out = dec.decode(tokens)
        return out["bb_pred"][0, 1:-1].detach().cpu().numpy().astype(np.float64)

    def _esm3_full(self, bb: Backbone) -> tuple[np.ndarray, int]:
        """ESM3 over the whole structure, in ``bb`` row order.

        Reuses :func:`recon_bench.adapters.esm3.chain_break_layout` rather than
        rebuilding it. Butt-joining chains instead cost ESM3 4.1 A of pocket
        Kabsch on CASP16's two-chain samples and produced every one of its
        apparent outliers -- rewriting that layout here is how the same bug
        comes back under a different model name.
        """
        coords, residue_index, is_residue, order = chain_break_layout(bb)
        pred = self._esm3_round_trip(
            coords, residue_index, np.flatnonzero(~is_residue)
        )[is_residue]
        # ``order`` is the bb row each ESM3 row came from; invert it so callers
        # can index by bb row.
        back = np.empty(len(bb), dtype=np.int64)
        back[order] = np.arange(order.size)
        return pred[back], int(is_residue.sum() + (~is_residue).sum())

    def _esm3_pocket(self, coords: np.ndarray) -> tuple[np.ndarray, int]:
        """ESM3 over the pocket residues alone, renumbered 1..L.

        Kept for the ablation that shows why the default is ``full``: a pocket
        is discontiguous and this presents residues angstroms apart as chain
        neighbours.
        """
        n = coords.shape[0]
        pred = self._esm3_round_trip(
            coords, np.arange(1, n + 1, dtype=np.int64), np.empty(0, dtype=np.int64)
        )
        return pred, n

    def _confseq_ligand(self, ligand_sdf: Path):
        """Round-trip a ligand through ConfSeq.

        Returns ``(ref_ordered, rec_coords, elements, bonds, orders, n_tokens)``
        with ``ref`` reordered onto the decoded molecule's own atom order, the
        way :mod:`recon_bench.adapters.confseq` does it.
        """
        from rdkit import Chem

        cs = self._confseq
        ref = Chem.MolFromMolFile(str(ligand_sdf), removeHs=True, sanitize=True)
        if ref is None or ref.GetNumConformers() == 0:
            msg = "bad ligand mol"
            raise ValueError(msg)
        _in_smiles, confseq = cs.get_ConfSeq_pair_from_mol(cs.aug_mol(ref, 0))
        in_smiles = cs.replace_angle_brackets_with_line(confseq)
        rec_mol = cs.get_mol_from_ConfSeq_pair(in_smiles, confseq)
        rec_mol = Chem.MolFromMolBlock(
            cs.remove_degree_in_molblock(Chem.MolToMolBlock(rec_mol))
        )
        if rec_mol is None or rec_mol.GetNumConformers() == 0:
            msg = "confseq decode failed"
            raise ValueError(msg)
        match = ref.GetSubstructMatch(rec_mol)
        if len(match) != rec_mol.GetNumAtoms():
            msg = f"atom match failed ({len(match)} of {rec_mol.GetNumAtoms()})"
            raise ValueError(msg)
        ref_ordered = np.asarray(
            ref.GetConformer().GetPositions()[list(match)], dtype=np.float64
        )
        rec_coords = np.asarray(rec_mol.GetConformer().GetPositions(), dtype=np.float64)
        bonds, orders = [], []
        for bond in rec_mol.GetBonds():
            bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            orders.append(
                {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}.get(
                    str(bond.GetBondType()), 1
                )
            )
        elements = [a.GetSymbol() for a in rec_mol.GetAtoms()]
        return (
            ref_ordered,
            rec_coords,
            elements,
            bonds,
            orders,
            len(confseq.split()),
        )

    # -- the staple -------------------------------------------------------
    def reconstruct(self, sample: Sample) -> ReconResult:  # noqa: PLR0915
        from prolit.tokenizers.pose_budget import quantize_pose

        if sample.protein_pdb is None or sample.ligand_sdf is None:
            return ReconResult(
                self.name, sample.sample_id, ok=False, error="needs protein + ligand"
            )
        self.setup()
        try:
            lig_ref, lig_local, lig_elements, bonds, orders, n_lig_tok = (
                self._confseq_ligand(sample.ligand_sdf)
            )
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            return ReconResult(
                self.name, sample.sample_id, ok=False, error=f"confseq: {exc}"
            )

        bb = read_backbone(sample.protein_pdb, chain=sample.chain)
        d = np.linalg.norm(bb.ca[:, None, :] - lig_ref[None, :, :], axis=-1)
        pocket = np.flatnonzero(d.min(axis=1) <= self.pocket_radius)
        if pocket.size < 3:
            return ReconResult(
                self.name, sample.sample_id, ok=False, error="pocket has <3 residues"
            )

        try:
            ref_bb = bb.coords[pocket]
            if self.protein_scope == "pocket":
                rec_bb, n_prot_tok = self._esm3_pocket(ref_bb)
            else:
                full, n_prot_tok = self._esm3_full(bb)
                rec_bb = full[pocket]
        except (RuntimeError, ValueError) as exc:
            return ReconResult(
                self.name, sample.sample_id, ok=False, error=f"esm3: {exc}"
            )

        # ESM3 decodes into its own frame; the receptor is known in any real
        # use, so the pocket is put back on the reference by superposition. That
        # gives the baseline its placement for free and isolates the question
        # this row is about, which is the LIGAND's placement.
        ref_flat = ref_bb.reshape(-1, 3)
        rec_flat = rec_bb.reshape(-1, 3)
        rot_p, t_p = _kabsch(rec_flat, ref_flat)
        rec_flat = rec_flat @ rot_p.T + t_p

        # The ligand's true rigid transform, then quantized to the budget. The
        # box is the pocket's bounding cube, which is what a sender and a
        # receiver can both compute from the receptor alone.
        rot_l, _ = _kabsch(lig_local, lig_ref)
        centroid = lig_ref.mean(axis=0)
        box_origin = ref_flat.min(axis=0)
        box_size = float((ref_flat.max(axis=0) - box_origin).max())
        centroid_q, rot_q = quantize_pose(
            centroid, rot_l, box_origin, box_size, self.pose_bits, self.seed
        )
        lig_rec = (lig_local - lig_local.mean(axis=0)) @ rot_q.T + centroid_q

        prot_elements = ["N", "C", "C"] * len(pocket)
        n_pose_tok = 0 if self.pose_bits is None else -(-self.pose_bits // 13)
        total_bits = (
            n_prot_tok * ESM3_CODEBOOK_BITS
            + n_lig_tok * CONFSEQ_BITS
            + float(self.pose_bits or 0)
        )
        rate = {
            "bits_protein": ESM3_CODEBOOK_BITS,
            "bits_ligand": CONFSEQ_BITS,
            "pose_bits": float(self.pose_bits or 0),
            "ligand_frame": "local",
            "arm_label": (
                "ESM3 + ConfSeq"
                + ("" if self.pose_bits is None else f" + {self.pose_bits}b pose")
            ),
            "total_bits": total_bits,
            "n_tokens_protein": n_prot_tok,
            "n_tokens_ligand": n_lig_tok,
            "n_tokens_pose": n_pose_tok,
        }
        res_keys = [
            (str(bb.chain_ids[r]), int(bb.res_ids[r])) for r in pocket.tolist()
        ]
        modalities = [
            ModalityRecon(
                modality="protein_backbone",
                ref=ref_bb[:, 1, :],
                rec=rec_flat.reshape(-1, 3, 3)[:, 1, :],
                atom_kind="CA",
                n_residues=int(pocket.size),
                n_tokens=n_prot_tok,
                res_keys=res_keys,
                extra=dict(rate),
            ),
            ModalityRecon(
                modality="ligand",
                ref=lig_ref,
                rec=lig_rec,
                atom_kind="heavy",
                n_tokens=n_lig_tok + n_pose_tok,
                extra={
                    **rate,
                    "elements": lig_elements,
                    "bonds": bonds,
                    "bond_orders": orders,
                },
            ),
            # Stacked in one frame -- where a missing binding pose shows up.
            ModalityRecon(
                modality="complex",
                ref=np.vstack([ref_flat, lig_ref]),
                rec=np.vstack([rec_flat, lig_rec]),
                atom_kind="heavy",
                n_tokens=n_prot_tok + n_lig_tok + n_pose_tok,
                extra={
                    **rate,
                    "n_protein_rows": int(ref_flat.shape[0]),
                    "protein_elements": prot_elements,
                    "ligand_elements": lig_elements,
                },
            ),
        ]
        return ReconResult(self.name, sample.sample_id, modalities=modalities)
