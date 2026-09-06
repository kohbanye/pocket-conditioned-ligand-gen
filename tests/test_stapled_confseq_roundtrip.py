"""ConfSeq really does throw the pose away, and the pose channel really puts it back.

The stapled baseline rests on two claims that are easy to assert and easy to get
wrong in code: that a ligand-only tokenizer's output carries no placement, and
that the quantized channel restores it to within the budget's resolution. Both
are measured here on real ligands rather than argued.

Skipped where ConfSeq's dependencies are absent (it needs Indigo); the
reconstruction benchmark's environment always has them, and the corpus builders'
environment gets them from the ``stapled`` dependency group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prolit.tokenizers.pose_budget import SemanticPoseCodec
from prolit.tokenizers.stapled import (
    confseq_encode,
    place,
    pocket_box,
    pose_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFSEQ_REPO = REPO_ROOT / "third_party" / "ConfSeq"
# mol2, not sdf. CASF ships both, and ConfSeq decodes 285/285 of the mol2
# ligands against 0/190 of the sdf ones -- the sdf files are malformed enough
# that RDKit also drops 92 of 285 outright. See
# docs/notes/2026-09-05_confseq_coverage_was_the_sdf.md.
LIGANDS = [
    p
    for p in sorted(
        (REPO_ROOT / "data" / "casf2016" / "coreset").glob("*/*_ligand.mol2")
    )
    if "opt" not in p.name
]


def _mols(n: int) -> list:
    pytest.importorskip("indigo")
    if not CONFSEQ_REPO.exists() or not LIGANDS:
        pytest.skip("ConfSeq checkout or CASF ligands not present")
    from rdkit import Chem, RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    out = []
    for path in LIGANDS:
        mol = Chem.MolFromMol2File(str(path), removeHs=True, sanitize=True)
        if mol is None or mol.GetNumConformers() == 0:
            continue
        out.append(mol)
        if len(out) == n:
            break
    if not out:
        pytest.skip("no parsable CASF ligands")
    return out


def _encodings(n: int) -> list:
    got = []
    for mol in _mols(n * 3):
        enc = confseq_encode(mol, CONFSEQ_REPO)
        if enc is not None:
            got.append(enc)
        if len(got) == n:
            break
    if not got:
        pytest.skip("ConfSeq encoded none of the sampled ligands")
    return got


def test_encoding_aligns_the_two_frames() -> None:
    """Both coordinate sets come back in the decoded molecule's atom order."""
    for enc in _encodings(3):
        assert enc.own.shape == enc.reference.shape
        assert enc.own.shape[0] > 0
        assert len(enc.tokens) > 0


def test_confseq_carries_no_placement() -> None:
    """The decoded molecule is nowhere near where the ligand actually sits.

    This is the premise of the whole baseline. If it ever became false -- say
    ConfSeq started emitting absolute coordinates -- the pose channel would be
    handing over information the tokens already had, and every rate claim built
    on it would be wrong.
    """
    for enc in _encodings(3):
        displacement = np.linalg.norm(
            enc.own.mean(axis=0) - enc.reference.mean(axis=0)
        )
        internal = np.linalg.norm(
            enc.own - enc.own.mean(axis=0), axis=1
        ).mean()
        assert displacement > internal, (
            "decoded ligand sits within its own radius of the true pose; "
            "ConfSeq may no longer be frame-free"
        )


def test_pose_channel_restores_the_placement() -> None:
    """With the channel, the molecule lands back on the ligand.

    The residual is bounded by the quantizer, not by ConfSeq: internal shape is
    whatever ConfSeq reproduced and is measured separately (bond MAE 0.040 A on
    PoseBusters). What this checks is that nothing in the frame bookkeeping is
    transposed -- an inverted rotation still gives a plausible-looking molecule
    in the right neighbourhood, which is exactly why it needs a number.
    """
    for enc in _encodings(3):
        backbone = enc.reference + np.array([25.0, 0.0, 0.0])  # a stand-in pocket
        backbone = np.vstack([backbone, enc.reference - 25.0])
        codec = SemanticPoseCodec()
        code = pose_of(enc.reference, enc.own, backbone, codec)
        placed = place(enc.own, code, backbone, codec)
        centroid_err = np.linalg.norm(
            placed.mean(axis=0) - enc.reference.mean(axis=0)
        )
        # The channel's own resolution, not ConfSeq's internal error: the
        # centroid is exactly what the translation tokens carry.
        assert centroid_err < 0.1, f"{centroid_err:.3f} A of placement error"


def test_zero_channel_is_the_middle_of_the_box() -> None:
    """No channel means no information -- and it must look like it."""
    enc = _encodings(1)[0]
    backbone = np.vstack([enc.reference + 25.0, enc.reference - 25.0])
    origin, size = pocket_box(backbone)
    placed = place(enc.own, None, backbone)
    assert np.allclose(placed.mean(axis=0), origin + size / 2.0)
