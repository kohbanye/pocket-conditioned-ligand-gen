"""Token-Mol 1.0 ligand reconstruction (in-process, no neural weights needed).

Token-Mol tokenizes a molecule as SMILES + the dihedral angles of its rotatable
bonds (discretized to 0.01 rad). Reconstruction is therefore a *tokenizer*
round-trip — no GPT-2 weights required: embed a conformer from the SMILES
(RDKit) and set the rotatable-bond torsions to the reference values (Token-Mol's
own von-Mises best-fit). Bond lengths/angles, rings, and non-rotatable geometry
come from the RDKit embedding, so the RMSD vs the original captures everything
the torsion representation cannot store.

Ligand-only, comparable to the own model's ligand modality. Uses Token-Mol's
``utils/standardization.py`` from the submodule.
"""

from __future__ import annotations

import sys

import numpy as np

from recon_bench import paths
from recon_bench.adapters.base import ReconstructionModel
from recon_bench.types import ModalityRecon, ReconResult, Sample


class TokenMolAdapter(ReconstructionModel):
    name = "token_mol"
    can_protein = False
    can_ligand = True

    def __init__(self, seed: int = 42, **_: object) -> None:
        self.seed = seed
        self._fns = None

    def setup(self) -> None:
        if self._fns is not None:
            return
        if str(paths.TOKEN_MOL_REPO) not in sys.path:
            sys.path.insert(0, str(paths.TOKEN_MOL_REPO))
        from rdkit import RDLogger
        from utils.standardization import (  # type: ignore
            apply_changes,
            get_dihedral_vonMises,
            get_torsion_angles,
        )

        RDLogger.DisableLog("rdApp.*")
        self._fns = (get_torsion_angles, get_dihedral_vonMises, apply_changes)

    def reconstruct(self, sample: Sample) -> ReconResult:
        if sample.ligand_sdf is None:
            return ReconResult(self.name, sample.sample_id, ok=False, error="no ligand_sdf")
        self.setup()
        from rdkit import Chem
        from rdkit.Chem import AllChem

        get_torsion_angles, get_dihedral_vonMises, apply_changes = self._fns

        ref = Chem.MolFromMolFile(str(sample.ligand_sdf), removeHs=True, sanitize=True)
        if ref is None or ref.GetNumConformers() == 0:
            return ReconResult(self.name, sample.sample_id, ok=False, error="bad ligand mol")

        smiles = Chem.MolToSmiles(ref)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ReconResult(self.name, sample.sample_id, ok=False, error="smiles re-parse failed")
        molh = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(molh, randomSeed=self.seed) != 0:
            return ReconResult(self.name, sample.sample_id, ok=False, error="embed failed")
        mol = Chem.RemoveHs(molh)

        match = ref.GetSubstructMatch(mol)  # match[i] = ref atom for reconstructed atom i
        if len(match) != mol.GetNumAtoms():
            return ReconResult(self.name, sample.sample_id, ok=False, error="atom match failed")
        ref_re = Chem.RenumberAtoms(ref, list(match))  # align ref order to mol order

        rotatable = get_torsion_angles(mol)
        ref_pos = ref_re.GetConformer().GetPositions()
        if rotatable:
            dihedrals = np.array(
                [get_dihedral_vonMises(mol, mol.GetConformer(0), r, ref_pos) for r in rotatable]
            )
            mol = apply_changes(mol, dihedrals, rotatable, 0)

        rec = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
        ref_coords = np.asarray(ref_pos, dtype=np.float64)
        modality = ModalityRecon(
            modality="ligand",
            ref=ref_coords,
            rec=rec,
            atom_kind="heavy",
            n_tokens=int(len(rotatable)),  # torsion tokens stored (per rotatable bond)
            extra={"n_rotatable": len(rotatable)},
        )
        return ReconResult(self.name, sample.sample_id, modalities=[modality])
