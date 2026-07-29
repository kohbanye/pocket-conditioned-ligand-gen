"""ConfSeq ligand reconstruction (in-process, no neural weights needed).

ConfSeq (Xiong et al., *Nature Machine Intelligence* 2026) is a **conformation
description language**: SMILES augmented with discretized dihedrals, bond angles
and a pseudo-chirality descriptor. It is rule-based, so the round trip needs no
trained weights at all -- which also means there is no train/test contamination
to argue about.

The paper states the representation is SE(3)-invariant, and the decoder makes
that concrete: it rebuilds the molecule from the SMILES and *re-embeds* it, then
applies the stored internal coordinates. Nothing in the token string says where
the ligand sat in its pocket. That places ConfSeq in the same family as our
``localframe_*`` arms and Token-Mol: excellent internal geometry, no pose, so a
rigid transform has to be transmitted on the side before any interface metric
means anything.

Two consequences show up directly in the results table:

* ``kabsch_rmsd`` measures what ConfSeq actually encodes (internal shape).
* ``rmsd`` (no superposition) is meaningless-by-construction for it, and will be
  tens of Angstroms -- that gap *is* the pose information the representation
  does not carry.

Rate is also not comparable atom-for-atom: ConfSeq spends tokens per SMILES
symbol and per rotatable bond, not per atom, so the ``n_tokens`` column must be
read alongside it rather than the RMSD alone.
"""

from __future__ import annotations

import sys

import numpy as np

from plbench import paths
from plbench.adapters.base import ReconstructionModel
from plbench.types import ModalityRecon, ReconResult, Sample


class ConfSeqAdapter(ReconstructionModel):
    """Encode a ligand to a ConfSeq string and decode it back to 3D."""

    name = "confseq"
    can_protein = False
    can_ligand = True

    def __init__(self, aug_mode: int = 0, **_: object) -> None:
        # aug_mode 0 = rooted-SMILES augmentation at atom 0 (the deterministic
        # default in the authors' demo); higher modes randomize the root, which
        # would make the benchmark irreproducible run to run.
        self.aug_mode = aug_mode
        self._mod = None

    def setup(self) -> None:
        if self._mod is not None:
            return
        demo = paths.CONFSEQ_REPO / "demo"
        if str(demo) not in sys.path:
            sys.path.insert(0, str(demo))
        import ConfSeq  # type: ignore[import-not-found]
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        self._mod = ConfSeq

    def reconstruct(self, sample: Sample) -> ReconResult:
        if sample.ligand_sdf is None:
            return ReconResult(self.name, sample.sample_id, ok=False, error="no ligand_sdf")
        self.setup()
        from rdkit import Chem

        cs = self._mod
        ref = Chem.MolFromMolFile(str(sample.ligand_sdf), removeHs=True, sanitize=True)
        if ref is None or ref.GetNumConformers() == 0:
            return ReconResult(self.name, sample.sample_id, ok=False, error="bad ligand mol")

        # --- encode -> ConfSeq string ------------------------------------
        ran = cs.aug_mol(ref, self.aug_mode)
        _in_smiles, confseq = cs.get_ConfSeq_pair_from_mol(ran)

        # --- decode -> fresh 3D conformer --------------------------------
        # The decoder only ever sees the token string, so this is a true round
        # trip: it re-derives connectivity from the SMILES half and geometry
        # from the angle tokens.
        in_smiles = cs.replace_angle_brackets_with_line(confseq)
        rec_mol = cs.get_mol_from_ConfSeq_pair(in_smiles, confseq)
        rec_mol = Chem.MolFromMolBlock(
            cs.remove_degree_in_molblock(Chem.MolToMolBlock(rec_mol))
        )
        if rec_mol is None or rec_mol.GetNumConformers() == 0:
            return ReconResult(self.name, sample.sample_id, ok=False, error="decode failed")

        # The decoded molecule is built from SMILES, so its atom order is its
        # own; match it back onto the reference before comparing coordinates.
        match = ref.GetSubstructMatch(rec_mol)
        if len(match) != rec_mol.GetNumAtoms():
            return ReconResult(
                self.name, sample.sample_id, ok=False,
                error=f"atom match failed ({len(match)} of {rec_mol.GetNumAtoms()})",
            )
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

        modality = ModalityRecon(
            modality="ligand",
            ref=ref_ordered,
            rec=rec_coords,
            atom_kind="heavy",
            n_tokens=int(len(confseq.split())),
            extra={
                "elements": [a.GetSymbol() for a in rec_mol.GetAtoms()],
                "bonds": bonds,
                "bond_orders": orders,
                "ligand_frame": "local",
                "confseq_len": len(confseq.split()),
            },
        )
        return ReconResult(self.name, sample.sample_id, modalities=[modality])
