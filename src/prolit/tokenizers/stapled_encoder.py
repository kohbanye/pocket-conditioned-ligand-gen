"""Encoding a complex the way the stapled baseline sees it.

The counterpart of :class:`prolit.tokenizers.pose_encoder.PoseEncoder`, and
deliberately the same shape: set the pocket up once, then encode pose after pose
against it. Every corpus builder needs this -- decoys for the rescoring head,
CrossDocked poses for the generator, BioLiP and PLINDER complexes for
pretraining -- so it lives here rather than beside any one of them.

Three differences from ``PoseEncoder``, and each is the baseline's premise
rather than an implementation detail:

* The pocket is not encoded here at all. ESM3's package pins a fork of
  ``transformers``, so its structure tokens are precomputed per receptor by
  ``pipelines/corpora/esm3_structure_tokens.py`` and looked up from
  :class:`prolit.data.esm3_tokens.Esm3TokenCache`. Which residues to look up is
  still ProLIT's own pocket extraction, so both arms condition on **the same
  residues** of the same receptor.
* The ligand is encoded in its own frame by ConfSeq, which is rule-based and
  needs no weights -- and carries no placement.
* The placement is therefore sent separately, as the four hierarchical tokens
  of :class:`~prolit.tokenizers.pose_budget.SemanticPoseCodec`. That channel is
  the thing being priced, and its own resolution (~0.06 A) is an order of
  magnitude below what the downstream tasks resolve, so a result is about the
  representation rather than about this quantizer.

A pose this cannot represent is skipped and counted, never silently replaced by
its input coordinates: crediting the tokens with information they do not carry
is exactly the error the whole comparison exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from prolit.tokenizers.protein import (
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates_from_text,
)
from prolit.tokenizers.stapled import (
    StapledVocab,
    confseq_encode,
    pose_of,
)

if TYPE_CHECKING:
    from prolit.config import PocketExtractionConfig
    from prolit.data.esm3_tokens import Esm3TokenCache

__all__ = [
    "StapledEncoder",
    "StapledPocket",
    "build_vocab",
    "heavy_atom_mol",
    "merge_vocabs",
]

#: The atoms ESM3 reconstructs. The placement grid is built from these so that a
#: receiver holding only the pocket block can derive the identical box.
_BACKBONE_ATOMS = ("N", "CA", "C")


@dataclass(frozen=True)
class StapledPocket:
    """What one receptor contributes, computed once and reused for every pose."""

    codes: list[int]
    backbone: np.ndarray
    residue_ids: list[tuple[str, int]]


def heavy_atom_mol(atoms: list, bonds: list, heavy_idx: np.ndarray) -> Any | None:  # noqa: ANN401
    """RDKit mol over the heavy atoms only, carrying their coordinates.

    ConfSeq reasons about a heavy-atom molecular graph; handing it the explicit
    hydrogens a PDB ligand record carries changes the SMILES it writes and the
    dihedrals it stores. ProLIT drops hydrogens too, so this keeps the two arms
    describing the same molecule.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Geometry import Point3D  # noqa: PLC0415

    bt = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    keep = {int(i): k for k, i in enumerate(heavy_idx)}
    rw = Chem.RWMol()
    for i in heavy_idx:
        try:
            rw.AddAtom(Chem.Atom(atoms[int(i)][0]))
        except Exception:  # noqa: BLE001
            rw.AddAtom(Chem.Atom("C"))
    for i, j, t in bonds:
        a, b = keep.get(int(i)), keep.get(int(j))
        if a is None or b is None or a == b:
            continue
        try:
            rw.AddBond(a, b, bt.get(t, Chem.BondType.SINGLE))
        except Exception:  # noqa: BLE001, S112
            continue
    mol = rw.GetMol()
    conf = Chem.Conformer(len(heavy_idx))
    for k, i in enumerate(heavy_idx):
        a = atoms[int(i)]
        conf.SetAtomPosition(k, Point3D(float(a[1]), float(a[2]), float(a[3])))
    mol.AddConformer(conf)
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001
        # A BioLiP ligand record plus a CCD SMILES does not always give RDKit a
        # valence-clean molecule -- metal coordination written as covalent
        # bonds, a nitro group as N(=O)=O, a charge the record does not carry.
        # Full sanitization refuses those, and refusing costs the baseline 15%
        # of its poses, which would look like a limit of ConfSeq rather than of
        # this function. ConfSeq needs ring and aromaticity perception and
        # nothing else, so retry without the valence check.
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:  # noqa: BLE001
            return None
    return mol


@dataclass
class StapledEncoder:
    """Fixed-pocket encoder for the ESM3 + ConfSeq baseline."""

    cache: Esm3TokenCache | None
    confseq_repo: Path
    vocab: StapledVocab
    pocket_cfg: PocketExtractionConfig

    def setup_pocket(
        self,
        struct_id: str,
        protein_text: str,
        reference_heavy: np.ndarray,
    ) -> StapledPocket | None:
        """ProLIT's pocket, with ESM3's codes for its residues.

        Returns ``None`` when the receptor yields no pocket, or when the cache
        does not cover every one of its residues -- a partially covered pocket
        is a different pocket, not a shorter one.
        """
        precomp = precompute_pocket_atom_candidates_from_text(protein_text)
        pocket = extract_pocket_atoms_from_candidates(
            precomp, reference_heavy, self.pocket_cfg
        )
        if pocket is None or pocket.atom_coords.shape[0] == 0:
            return None
        if self.cache is None:
            msg = "setup_pocket needs an ESM3 cache; this encoder has none"
            raise RuntimeError(msg)
        codes = self.cache.pocket_codes(struct_id, list(pocket.residue_ids))
        if codes is None:
            return None
        rows = [
            i
            for i, name in enumerate(pocket.atom_names)
            if name.strip() in _BACKBONE_ATOMS
        ]
        if not rows:
            return None
        return StapledPocket(
            codes=codes,
            backbone=np.asarray(pocket.atom_coords[rows], dtype=np.float64),
            residue_ids=list(pocket.residue_ids),
        )

    def ligand_seq(
        self,
        pocket: StapledPocket,
        atoms: list,
        bonds: list,
        heavy_idx: np.ndarray,
    ) -> list[int] | None:
        """One pose as a token stream, or ``None`` if it cannot be written."""
        return self.ligand_seq_with_reason(pocket, atoms, bonds, heavy_idx)[0]

    def ligand_seq_with_reason(
        self,
        pocket: StapledPocket,
        atoms: list,
        bonds: list,
        heavy_idx: np.ndarray,
    ) -> tuple[list[int] | None, str]:
        """As :meth:`ligand_seq`, but say *why* when it fails.

        Two failures look identical in a corpus and mean opposite things. A
        molecule ConfSeq cannot describe is a real limit of the baseline and
        belongs in the paper. A molecule whose tokens are out of vocabulary is
        a **build error** -- the frozen alphabet was collected from too small a
        sample -- and silently drops good data: a vocabulary missing only
        ``#``, ``o``, ``s`` and ``r`` cost 63% of a GEOM sample here. Counting
        them together would have hidden that behind a plausible number.
        """
        mol = heavy_atom_mol(atoms, bonds, heavy_idx)
        if mol is None:
            return None, "rdkit"
        enc = confseq_encode(mol, self.confseq_repo)
        if enc is None:
            return None, "confseq"
        ids = self.vocab.confseq.ids(enc.tokens)
        if ids is None:
            return None, "oov"
        code = pose_of(enc.reference, enc.own, pocket.backbone, self.vocab.pose)
        return self.vocab.build_sequence(pocket.codes, ids, code), "ok"

    def missing_symbols(self, tokens: list[str]) -> list[str]:
        """Which of ``tokens`` the frozen vocabulary does not contain."""
        known = set(self.vocab.confseq.tokens)
        return sorted({t for t in tokens if t not in known})

    def ligand_only_seq(
        self, atoms: list, bonds: list, heavy_idx: np.ndarray
    ) -> list[int] | None:
        """A conformer with no pocket: empty ``<p></p>`` and no placement.

        GEOM is ligands alone, so there is nothing to be placed relative to.
        ProLIT's own corpus carries these as an empty pocket block too, which is
        what lets single-modality pretraining and complex fine-tuning share one
        format -- the stapled stream keeps that property.
        """
        mol = heavy_atom_mol(atoms, bonds, heavy_idx)
        if mol is None:
            return None
        enc = confseq_encode(mol, self.confseq_repo)
        if enc is None:
            return None
        ids = self.vocab.confseq.ids(enc.tokens)
        if ids is None:
            return None
        return self.vocab.build_sequence([], ids, None)

    def confseq_tokens(
        self, atoms: list, bonds: list, heavy_idx: np.ndarray
    ) -> list[str] | None:
        """The raw ConfSeq tokens of one pose, for building the vocabulary."""
        mol = heavy_atom_mol(atoms, bonds, heavy_idx)
        if mol is None:
            return None
        enc = confseq_encode(mol, self.confseq_repo)
        return None if enc is None else enc.tokens


def merge_vocabs(paths: list[Path]) -> StapledVocab:
    """One alphabet over several sources.

    Every corpus in a run has to share one ConfSeq vocabulary: the ids are baked
    into each stream and into the model's embedding table, so a per-source
    vocabulary would give one model several incompatible alphabets. GEOM is
    where the chemical diversity is and BioLiP is where the cofactors are, so
    the union of a pass over each is what gets frozen.

    The order is by name, not by frequency, because the counts come from
    different-sized passes and ranking one source's symbols above another's by
    an accident of sample size would make the ids depend on how long each pass
    happened to run.
    """
    from prolit.tokenizers.stapled import ConfSeqVocab  # noqa: PLC0415

    symbols: set[str] = set()
    for path in paths:
        symbols |= set(ConfSeqVocab.load(Path(path)).tokens)
    return build_vocab(dict.fromkeys(sorted(symbols), 1))


#: ConfSeq's non-angle alphabet, in full. Measured rather than assumed: over
#: GEOM and BioLiP its tokenizer splits every multi-character symbol, so ``Cl``
#: arrives as ``C`` then ``l``, ``[nH]`` as ``[``, ``n``, ``H``, ``]``, and
#: ``%10`` as ``%``, ``1``, ``0``. Nothing longer than one character survives
#: except the angle tokens, which means this half of the alphabet is closed and
#: can be enumerated instead of collected.
#:
#: Enumerating it is not a convenience. An out-of-vocabulary token does not
#: raise -- it drops the molecule -- so a vocabulary collected from too small a
#: sample silently shrinks the corpus and the model still trains and still
#: reports a loss. A pass over 396 GEOM conformers was missing ``#``, ``o``,
#: ``s`` and ``r``, and that alone cost 63% of a GEOM sample. With the alphabet
#: enumerated there is no sample size to get wrong and no data pass to
#: reproduce: the vocabulary is a constant.
_CONFSEQ_ALPHABET: tuple[str, ...] = (
    # Element letters. All of A-Z and a-z, which is 52 tokens against the ~20
    # that actually occur -- far cheaper than being wrong about which twenty.
    *(chr(c) for c in range(ord("A"), ord("Z") + 1)),
    *(chr(c) for c in range(ord("a"), ord("z") + 1)),
    # Ring closures and charges.
    *(str(d) for d in range(10)),
    # SMILES structure and bond symbols, plus ConfSeq's own markers.
    "(", ")", "[", "]", "{", "}", "|",
    ".", "=", "#", "-", "+", "\\", "/", ":", "~", "@", "?", ">", "*", "$", "%",
)


def build_vocab(counts: dict[str, int], angle_range: int = 180) -> StapledVocab:
    """Freeze a vocabulary that neither geometry nor common chemistry escapes.

    Both halves of ConfSeq's alphabet are closed sets, so both are enumerated:
    angles as ``<int degrees>`` over the full range, and the chemical symbols
    from :data:`_CONFSEQ_ALPHABET`. ``counts`` is therefore optional -- it is
    kept so a corpus pass can *check* the enumeration rather than depend on it,
    and any token it supplies that is not already enumerated is a signal that
    this assumption has broken and should be looked at, not quietly absorbed.
    """
    from prolit.tokenizers.stapled import ConfSeqVocab  # noqa: PLC0415

    full = dict(counts)
    for deg in range(-angle_range, angle_range + 1):
        full.setdefault(f"<{deg}>", 0)
    for sym in _CONFSEQ_ALPHABET:
        full.setdefault(sym, 0)
    return StapledVocab(confseq=ConfSeqVocab.from_counts(full))
