"""Validated ligand metadata preserves chemistry without changing legacy parsing."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from prolit.tokenizers.atom import LigandAtomDescriptor
from prolit.tokenizers.descriptor_schema import (
    ATOM_LAYOUT,
    LIGAND_CHARGE_TO_IDX,
    fields_by_name,
)
from prolit.tokenizers.ligand import (
    parse_ligand_pdb_text,
    validated_heavy_mol,
    validated_mol_from_mol2_text,
)


def pdb_for(smiles: str, *, explicit_h: bool = False) -> str:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    if not explicit_h:
        mol = Chem.RemoveHs(mol)
    return Chem.MolToPDBBlock(mol)


@pytest.mark.parametrize("smiles", ["C[N+](C)(C)C", "CC(=O)[O-]", "[NH3+]CC(=O)[O-]"])
def test_formal_charge_survives_descriptor_path(smiles: str) -> None:
    text = pdb_for(smiles)
    legacy = parse_ligand_pdb_text(text, smiles)
    item = parse_ligand_pdb_text(text, smiles, preserve_chemistry=True)
    assert item is not None
    assert legacy is not None
    assert set(legacy) == {"atoms", "bonds"}
    assert item["atoms"] == legacy["atoms"]
    assert item["bonds"] == legacy["bonds"]
    descriptor = LigandAtomDescriptor()
    frame = (np.zeros(3), np.eye(3))
    corrected, _, _ = descriptor.compute(
        item["atoms"], item["bonds"], frame, chemistry=item["chemistry"]
    )
    original, _, _ = descriptor.compute(legacy["atoms"], legacy["bonds"], frame)
    explicit_none, _, _ = descriptor.compute(
        legacy["atoms"], legacy["bonds"], frame, chemistry=None
    )
    np.testing.assert_array_equal(original, explicit_none)
    fields = fields_by_name(ATOM_LAYOUT)
    for name in ("coord", "knn_offsets", "element"):
        field = fields[name]
        np.testing.assert_array_equal(
            corrected[:, field.start : field.end], original[:, field.start : field.end]
        )
    restored = validated_heavy_mol(
        item["chemistry"], item["atoms"], item["bonds"], list(range(len(item["atoms"])))
    )
    charges = [LIGAND_CHARGE_TO_IDX[a.GetFormalCharge()] for a in restored.GetAtoms()]
    np.testing.assert_array_equal(corrected[:, fields["charge"].start], charges)
    assert any(a.GetFormalCharge() for a in restored.GetAtoms())


def test_explicit_hydrogens_and_reordered_heavy_atoms() -> None:
    item = parse_ligand_pdb_text(
        pdb_for("C[NH3+]", explicit_h=True), "C[NH3+]", preserve_chemistry=True
    )
    assert item is not None
    heavy = [i for i, atom in enumerate(item["atoms"]) if atom[0] != "H"]
    mol = validated_heavy_mol(
        item["chemistry"], item["atoms"], item["bonds"], heavy[::-1]
    )
    assert [a.GetSymbol() for a in mol.GetAtoms()] == [
        item["atoms"][i][0] for i in heavy[::-1]
    ]
    nitrogen = next(a for a in mol.GetAtoms() if a.GetSymbol() == "N")
    assert nitrogen.GetFormalCharge() == 1
    assert nitrogen.GetTotalNumHs() == 3


def test_stereo_survives_payload_round_trip() -> None:
    smiles = "N[C@@H](C)C(=O)O"
    item = parse_ligand_pdb_text(pdb_for(smiles), smiles, preserve_chemistry=True)
    assert item is not None
    mol = validated_heavy_mol(
        item["chemistry"], item["atoms"], item["bonds"], list(range(len(item["atoms"])))
    )
    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("template", [None, "CC", "not-a-smiles"])
def test_unvalidated_or_partial_template_fails_closed(template: str | None) -> None:
    with pytest.raises(ValueError, match="complete, sanitizable CCD"):
        parse_ligand_pdb_text(pdb_for("CCO"), template, preserve_chemistry=True)


def test_mismatched_payload_and_unknown_schema_rejected() -> None:
    item = parse_ligand_pdb_text(pdb_for("CCO"), "CCO", preserve_chemistry=True)
    assert item is not None
    heavy = list(range(len(item["atoms"])))
    with pytest.raises(ValueError, match="schema"):
        validated_heavy_mol(
            {**item["chemistry"], "schema": "future"},
            item["atoms"],
            item["bonds"],
            heavy,
        )
    with pytest.raises(ValueError, match="connectivity"):
        validated_heavy_mol(item["chemistry"], item["atoms"], [], heavy)
    with pytest.raises(ValueError, match="atom order"):
        validated_heavy_mol(
            item["chemistry"], item["atoms"][::-1], item["bonds"], heavy
        )


def test_dative_projection_preserves_directed_chemistry_and_descriptors() -> None:
    mol = Chem.MolFromSmiles("N->[Cu+2]<-N")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(i), 0.0, 0.0))
    mol.AddConformer(conf)
    atoms = [(a.GetSymbol(), float(a.GetIdx()), 0.0, 0.0) for a in mol.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), 1) for b in mol.GetBonds()]
    payload = {"schema": "validated_v1", "molblock": Chem.MolToMolBlock(mol)}
    restored = validated_heavy_mol(payload, atoms, bonds, [0, 1, 2])
    assert [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
        for b in restored.GetBonds()
    ] == [(0, 1, Chem.BondType.DATIVE), (2, 1, Chem.BondType.DATIVE)]
    assert [a.GetFormalCharge() for a in restored.GetAtoms()] == [0, 2, 0]
    reordered = validated_heavy_mol(payload, atoms, bonds, [2, 0, 1])
    assert [a.GetSymbol() for a in reordered.GetAtoms()] == ["N", "N", "Cu"]
    assert {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in reordered.GetBonds()} == {
        (1, 2),
        (0, 2),
    }
    desc, _, _ = LigandAtomDescriptor().compute(
        atoms, bonds, (np.zeros(3), np.eye(3)), chemistry=payload
    )
    assert np.isfinite(desc).all()
    charge = fields_by_name(ATOM_LAYOUT)["charge"].start
    np.testing.assert_array_equal(
        desc[:, charge], [LIGAND_CHARGE_TO_IDX[x] for x in [0, 2, 0]]
    )


def test_unsupported_bond_type_is_explicit_error() -> None:
    mol = Chem.RWMol(Chem.MolFromSmiles("CC"))
    mol.GetBondWithIdx(0).SetBondType(Chem.BondType.ZERO)
    payload = {"schema": "validated_v1", "molblock": Chem.MolToMolBlock(mol)}
    with pytest.raises(
        ValueError, match="Unsupported chemistry payload bond types: ZERO"
    ):
        validated_heavy_mol(
            payload, [("C", 0, 0, 0), ("C", 1, 0, 0)], [(0, 1, 1)], [0, 1]
        )


ETHANOL_MOL2 = """@<TRIPOS>MOLECULE
ethanol
 3 2 0 0 0
SMALL
NO_CHARGES
@<TRIPOS>ATOM
      1 C1      0.0000  0.0000  0.0000 C.3    1 LIG  0.0000
      2 C2      1.5000  0.0000  0.0000 C.3    1 LIG  0.0000
      3 O1      2.0000  1.2000  0.0000 O.3    1 LIG  0.0000
@<TRIPOS>BOND
     1    1    2 1
     2    2    3 1
"""


def _reference_from(text: str) -> dict:
    """The SDF-side mol dict the bench would score, taken from the same MOL2."""
    mol = Chem.MolFromMol2Block(text, sanitize=True, removeHs=False)
    assert mol is not None
    conf = mol.GetConformer()
    return {
        "atoms": [
            (
                a.GetSymbol(),
                conf.GetAtomPosition(a.GetIdx()).x,
                conf.GetAtomPosition(a.GetIdx()).y,
                conf.GetAtomPosition(a.GetIdx()).z,
            )
            for a in mol.GetAtoms()
        ],
        "bonds": [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), 1) for b in mol.GetBonds()
        ],
    }


def test_mol2_payload_is_accepted_and_preserves_the_mol2_chemistry() -> None:
    reference = _reference_from(ETHANOL_MOL2)
    item = validated_mol_from_mol2_text(reference, ETHANOL_MOL2)
    assert item is not None
    assert item["chemistry"]["schema"] == "validated_v1"
    assert item["chemistry"]["source"] == "mol2"
    heavy = [i for i, atom in enumerate(item["atoms"]) if atom[0] != "H"]
    restored = validated_heavy_mol(
        item["chemistry"], item["atoms"], item["bonds"], heavy
    )
    direct = Chem.MolFromMol2Block(ETHANOL_MOL2, sanitize=True, removeHs=False)
    assert [a.GetFormalCharge() for a in restored.GetAtoms()] == [
        a.GetFormalCharge() for a in direct.GetAtoms() if a.GetSymbol() != "H"
    ]
    descriptor, _, _ = LigandAtomDescriptor().compute(
        item["atoms"],
        item["bonds"],
        (np.zeros(3), np.eye(3)),
        chemistry=item["chemistry"],
    )
    assert np.isfinite(descriptor).all()


def test_a_different_pose_or_molecule_fails_closed() -> None:
    reference = _reference_from(ETHANOL_MOL2)
    moved = {
        "atoms": [(reference["atoms"][0][0], 9.0, 9.0, 9.0), *reference["atoms"][1:]],
        "bonds": reference["bonds"],
    }
    assert validated_mol_from_mol2_text(moved, ETHANOL_MOL2) is None
    shorter = {"atoms": reference["atoms"][:2], "bonds": []}
    assert validated_mol_from_mol2_text(shorter, ETHANOL_MOL2) is None
    swapped = {
        "atoms": [("N", *reference["atoms"][0][1:]), *reference["atoms"][1:]],
        "bonds": reference["bonds"],
    }
    assert validated_mol_from_mol2_text(swapped, ETHANOL_MOL2) is None


def test_unreadable_mol2_returns_none_rather_than_raising() -> None:
    reference = _reference_from(ETHANOL_MOL2)
    assert validated_mol_from_mol2_text(reference, "not a mol2 file") is None
    assert validated_mol_from_mol2_text(reference, "") is None
