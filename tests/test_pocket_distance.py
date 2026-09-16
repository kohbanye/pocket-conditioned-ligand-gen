"""Pocket selection must retain a contacting side chain despite a distant CA."""

from types import SimpleNamespace

import numpy as np
import pytest

from prolit.config import PocketExtractionConfig
from prolit.tokenizers.protein import (
    PrecomputedPocketAtoms,
    extract_pocket_atoms_from_candidates,
)


def _candidates() -> PrecomputedPocketAtoms:
    ca = np.array([[9.0, 0, 0], [5.0, 0, 0], [7.0, 0, 0]], dtype=np.float32)
    return PrecomputedPocketAtoms(
        ca_coords=ca,
        chain_ids=["A"] * 3,
        residue_indices=[1, 2, 3],
        residue_names=["LYS", "ALA", "ARG"],
        residue_atoms=[
            [("CA", "C", ca[0]), ("NZ", "N", np.array([2.0, 0, 0]))],
            [("CA", "C", ca[1])],
            [("CA", "C", ca[2]), ("NH1", "N", np.array([3.0, 0, 0]))],
        ],
    )


def test_heavy_distance_recovers_contact_and_caps_nearest_residues() -> None:
    ligand = np.zeros((1, 3))
    legacy = extract_pocket_atoms_from_candidates(
        _candidates(), ligand, PocketExtractionConfig()
    )
    assert legacy.residue_ids == [("A", 2), ("A", 3)]
    heavy = extract_pocket_atoms_from_candidates(
        _candidates(),
        ligand,
        PocketExtractionConfig(pocket_distance="heavy", max_residues=2),
    )
    assert heavy.residue_ids == [("A", 1), ("A", 3)]
    assert "NZ" in heavy.atom_names
    ordered = extract_pocket_atoms_from_candidates(
        _candidates(),
        ligand,
        PocketExtractionConfig(
            pocket_distance="heavy", max_residues=2, pocket_order="distance"
        ),
    )
    assert ordered.residue_ids == [("A", 3), ("A", 1)]


def test_legacy_config_without_new_field_and_invalid_metric() -> None:
    legacy = SimpleNamespace(distance_cutoff=8.0, max_residues=50)
    result = extract_pocket_atoms_from_candidates(
        _candidates(), np.zeros((1, 3)), legacy
    )
    assert result.residue_ids == [("A", 2), ("A", 3)]
    with pytest.raises(ValueError, match="pocket_distance"):
        extract_pocket_atoms_from_candidates(
            _candidates(),
            np.zeros((1, 3)),
            PocketExtractionConfig(pocket_distance="typo"),
        )
