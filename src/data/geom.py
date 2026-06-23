"""GEOM conformer ingestion for ligand-only LM pretraining.

Reads the GEOM ``rdkit_folder`` distribution (Axelrod & Gomez-Bombarelli,
*Scientific Data* 2022; https://github.com/learningmatter-mit/geom) and yields
per-conformer atom/bond dicts in the exact shape produced by
:func:`src.tokenizers.ligand.parse_sdf` (``{"atoms": [(elem, x, y, z), ...],
"bonds": [(a1, a2, bond_type), ...]}``), so the existing
:class:`~src.tokenizers.ligand.LigandDescriptor` pipeline can consume them
unchanged.

Directory layout after extracting ``rdkit_folder.tar.gz``::

    <geom_root>/
        summary_drugs.json   # {smiles: {"pickle_path": "drugs/0.pickle", ...}}
        summary_qm9.json
        drugs/*.pickle       # one molecule each: {"conformers": [{"rd_mol": Mol,
        qm9/*.pickle         #   "boltzmannweight": float, ...}, ...], ...}

The train/val/test split is assigned at the **molecule** level (hash of the
canonical SMILES) so conformers of the same molecule never leak across splits.
Conformers within a molecule are ranked by Boltzmann weight and the top
``max_confs_per_mol`` are kept (the low-energy conformers carry the signal we
want the LM to learn).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

# RDKit bond type -> SDF integer code used by parse_sdf / _build_rdkit_mol.
_BOND_TYPE_CODE = {
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
}

SUBSET_SUMMARY = {
    "drugs": "summary_drugs.json",
    "qm9": "summary_qm9.json",
}


@dataclass(frozen=True)
class GeomMolRef:
    """A molecule entry from a GEOM summary file."""

    smiles: str
    pickle_path: str  # relative to geom_root
    split: str  # "train" | "val" | "test"


def assign_split(
    smiles: str,
    val_frac: float,
    test_frac: float,
    seed: int = 0,
) -> str:
    """Deterministically map a SMILES to a split via a stable hash.

    Molecule-level (not conformer-level) so all conformers of one molecule land
    in the same split. ``blake2b`` keeps this stable across processes/runs
    (Python's built-in ``hash`` is salted per-process).
    """
    digest = hashlib.blake2b(
        f"{seed}:{smiles}".encode(),
        digest_size=8,
    ).hexdigest()
    frac = int(digest, 16) / float(1 << 64)  # in [0, 1)
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "val"
    return "train"


def load_geom_refs(  # noqa: PLR0913
    geom_root: Path,
    subsets: list[str],
    *,
    val_frac: float,
    test_frac: float,
    seed: int = 0,
    max_mols: int | None = None,
) -> list[GeomMolRef]:
    """Read GEOM summary JSON(s) and return molecule refs with split labels.

    Entries without a ``pickle_path`` (molecules GEOM could not generate
    conformers for) are skipped. ``max_mols`` truncates for smoke tests.
    """
    refs: list[GeomMolRef] = []
    for subset in subsets:
        summary_name = SUBSET_SUMMARY.get(subset)
        if summary_name is None:
            msg = (
                f"Unknown GEOM subset {subset!r}; "
                f"expected one of {list(SUBSET_SUMMARY)}"
            )
            raise ValueError(msg)
        summary_path = geom_root / summary_name
        if not summary_path.exists():
            msg = f"GEOM summary not found: {summary_path}"
            raise FileNotFoundError(msg)
        summary: dict[str, dict] = json.loads(summary_path.read_text())
        kept = 0
        for smiles, info in summary.items():
            pickle_path = info.get("pickle_path")
            if not pickle_path:
                continue
            refs.append(
                GeomMolRef(
                    smiles=smiles,
                    pickle_path=pickle_path,
                    split=assign_split(smiles, val_frac, test_frac, seed),
                )
            )
            kept += 1
            if max_mols is not None and len(refs) >= max_mols:
                break
        logger.info("Loaded %d molecules from %s", kept, summary_path.name)
        if max_mols is not None and len(refs) >= max_mols:
            break
    return refs


def _rd_mol_to_atoms_bonds(rd_mol: object) -> dict | None:
    """Convert one RDKit conformer Mol to a parse_sdf-style atoms/bonds dict.

    Hydrogens are kept here (GEOM mols carry explicit H); the downstream
    :class:`LigandDescriptor` drops them and remaps bonds, exactly as it does
    for CrossDocked SDF input.
    """
    try:
        conf = rd_mol.GetConformer()  # type: ignore[attr-defined]
    except (ValueError, RuntimeError):
        return None

    atoms: list[tuple[str, float, float, float]] = []
    for atom in rd_mol.GetAtoms():  # type: ignore[attr-defined]
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append((atom.GetSymbol(), float(pos.x), float(pos.y), float(pos.z)))

    bonds: list[tuple[int, int, int]] = []
    for bond in rd_mol.GetBonds():  # type: ignore[attr-defined]
        code = _BOND_TYPE_CODE.get(str(bond.GetBondType()), 1)
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), code))

    if not atoms:
        return None
    return {"atoms": atoms, "bonds": bonds}


def _conformers_from_mol_dict(mol_dict: dict, max_confs_per_mol: int) -> Iterator[dict]:
    """Yield up to ``max_confs_per_mol`` parsed conformer dicts from a GEOM mol.

    Conformers are ranked by descending Boltzmann weight (most-populated first).
    """
    conformers = mol_dict.get("conformers") or []
    conformers = sorted(
        conformers,
        key=lambda c: c.get("boltzmannweight", 0.0),
        reverse=True,
    )
    emitted = 0
    for conf in conformers:
        if emitted >= max_confs_per_mol:
            break
        rd_mol = conf.get("rd_mol")
        if rd_mol is None:
            continue
        parsed = _rd_mol_to_atoms_bonds(rd_mol)
        if parsed is None:
            continue
        yield parsed
        emitted += 1


def iter_mol_conformers(
    geom_root: Path,
    ref: GeomMolRef,
    max_confs_per_mol: int,
) -> Iterator[dict]:
    """Yield parsed conformer dicts for one molecule from an extracted pickle."""
    path = geom_root / ref.pickle_path
    try:
        with path.open("rb") as fh:
            mol_dict = pickle.load(fh)  # noqa: S301
    except Exception:
        logger.exception("Failed to read GEOM pickle %s", path)
        return
    yield from _conformers_from_mol_dict(mol_dict, max_confs_per_mol)


def iter_geom_tar_conformers(  # noqa: PLR0913
    tar_path: Path,
    subsets: list[str],
    *,
    val_frac: float,
    test_frac: float,
    seed: int = 0,
    max_confs_per_mol: int,
    max_mols: int | None = None,
) -> Iterator[tuple[str, dict]]:
    """Stream ``(split, conformer_dict)`` straight from ``rdkit_folder.tar.gz``.

    Inode-safe ingestion: the GEOM ``rdkit_folder`` is ~440k per-molecule
    pickles, so we never extract it. ``tarfile`` opens the gzip tar in
    streaming mode (``r|gz``) and we read each molecule pickle's bytes in
    sequence. The molecule-level split is derived from each pickle's SMILES,
    so no summary JSON is needed.

    Only members whose path contains a requested ``subset`` directory
    (``drugs`` / ``qm9``) and ends in ``.pickle`` are processed.

    Note: Dataverse serves ``rdkit_folder.tar.gz`` *decompressed* (a plain
    POSIX tar despite the name), so we open with ``r|*`` (transparent
    compression detection) to handle both the plain tar and a genuine gzip.
    """
    import tarfile  # noqa: PLC0415

    subset_set = set(subsets)
    n_mols = 0
    with tarfile.open(tar_path, "r|*") as tar:
        try:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".pickle"):
                    continue
                segments = member.name.split("/")
                if not subset_set.intersection(segments):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                try:
                    mol_dict = pickle.loads(handle.read())  # noqa: S301
                except Exception:
                    logger.exception("Failed to read tar member %s", member.name)
                    continue
                smiles = mol_dict.get("smiles") or member.name
                split = assign_split(smiles, val_frac, test_frac, seed)
                for parsed in _conformers_from_mol_dict(mol_dict, max_confs_per_mol):
                    yield split, parsed
                n_mols += 1
                if max_mols is not None and n_mols >= max_mols:
                    break
        except tarfile.ReadError:
            # A truncated / still-downloading tar raises here mid-stream. Surface
            # it loudly (do NOT silently undercount) but keep what we streamed.
            logger.exception(
                "Tar stream ended early after %d molecules -- is the download "
                "complete? Treating as end-of-archive.",
                n_mols,
            )
