"""Which complexes must never enter training, and where that list comes from.

Three evaluation sets sit downstream of the tokenizer, and each contributes PDB
entries that a training corpus can silently contain:

* **CrossDocked fold-0 test** -- the split the generation table reports on. The
  atom DataModule already honours it *as a split*, but a corpus built from
  another source (BioLiP2, PLINDER) has no notion of CrossDocked folds and has
  to exclude those PDB ids by hand.
* **CASF-2016 core set** -- pose rescoring and affinity. This one is not
  hypothetical: 169 of the 285 core-set entries appear in the CrossDocked
  manifest and 25,345 of those rows are labelled ``fold0 == train``, so a
  tokenizer trained on the plain fold split has seen them.
* **sbdd-bench targets** -- the generation table. Two lists, because the table
  grew: :data:`SBDD_BENCH_PDBS` is the original three targets, and
  :func:`sbdd_bench_pockets` is the 100-target set it became. The second is
  keyed by CrossDocked pocket rather than PDB id, and is the one that matters:
  ProLIT's fold split disagrees with the split those targets came from on 54 of
  93 pockets.

Exclusion is by four-character PDB id, which is the only key the three sets and
the CrossDocked manifest share. That is coarser than per-complex matching (it
drops every pose of a receptor, not just the evaluated one), and deliberately
so: the leak we care about is the tokenizer having seen the receptor's geometry.

Both ``prolit`` (the VQ-VAE DataModule) and the corpus builders under
``pipelines/`` need this, so it lives here rather than in either of them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: sbdd-bench's three generation targets.
SBDD_BENCH_PDBS: frozenset[str] = frozenset({"1iep", "2ity", "3pbl"})

#: Default location of the generation benchmark's evaluation POCKETS, one
#: CrossDocked ``complex_dir`` per line. Pockets, not PDB ids: the SBDD
#: benchmark evaluates on 100 CrossDocked pockets and the manifest keys its
#: split on ``complex_dir``, so a PDB-id list cannot express the exclusion.
#:
#: This exists because ProLIT's ``cdonly_fold0`` split and the split those 100
#: targets come from are DIFFERENT splits that overlap: 54 of the 93 distinct
#: evaluation pockets are labelled ``train`` in the manifest, so a corpus built
#: on fold0-train contains half the generation benchmark. Excluding them costs
#: 2.6% of the training poses.
SBDD_BENCH_POCKET_LIST = Path("data/sbdd_bench_pockets.txt")


def sbdd_bench_pockets(path: Path = SBDD_BENCH_POCKET_LIST) -> set[str]:
    """The generation benchmark's evaluation pockets, as ``complex_dir`` names."""
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().split() if line.strip()}


#: Default location of the CASF-2016 core-set id list (one id per line).
DEFAULT_CASF_LIST = Path("data/casf2016_pdbs.txt")

#: Default location of the CrossDocked manifest.
DEFAULT_CD_MANIFEST = Path("data/hub_cache/repo/manifest.parquet")

_PDB_ID = re.compile(r"^([0-9a-zA-Z]{4})_")


def pdb_id_from_receptor(receptor_pdb: str) -> str | None:
    """``"2bq0_A_rec.pdb"`` -> ``"2bq0"``; ``None`` when the name has no id."""
    m = _PDB_ID.match(receptor_pdb)
    return m.group(1).lower() if m else None


def casf_pdbs(path: Path = DEFAULT_CASF_LIST) -> set[str]:
    """CASF-2016 core-set PDB ids, or an empty set if the list is absent."""
    if not path.exists():
        logger.warning("CASF id list not found at %s; excluding nothing", path)
        return set()
    return {tok.lower() for tok in path.read_text().split() if tok.strip()}


def crossdocked_test_pdbs(
    manifest: Path = DEFAULT_CD_MANIFEST,
    fold: int = 0,
    source_type: str = "cdonly",
) -> set[str]:
    """PDB ids on the CrossDocked test side of ``fold``."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    col = f"{source_type}_fold{fold}"
    table = pq.read_table(manifest, columns=["receptor_pdb", "source_type", col])
    df = table.to_pandas()
    df = df[(df["source_type"] == source_type) & (df[col] == "test")]
    ids = (df["receptor_pdb"].map(pdb_id_from_receptor)).dropna()
    return set(ids)


def sbdd_bench_receptor_pdbs(
    manifest: Path = DEFAULT_CD_MANIFEST,
    pockets: set[str] | None = None,
) -> set[str]:
    """PDB ids of the receptors behind the generation benchmark's pockets.

    :data:`SBDD_BENCH_PDBS` names three ids because the benchmark once had three
    targets. It now has 97, drawn from 104 receptors, and those ids are what a
    corpus keyed by PDB id -- BioLiP2, PLINDER -- has to exclude. Pocket names
    cannot do that job: a ``complex_dir`` like ``LMBL1_HUMAN_198_526_0`` carries
    no PDB id at all, and the one target whose name happens to embed one
    (``2pqw``) is the only one a name-based list would have caught.

    The mapping therefore comes from the manifest, which is the only place that
    joins a pocket to its receptor files. Returns an empty set when the manifest
    is absent, like the other loaders here.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    manifest = Path(manifest)
    if not manifest.exists():
        logger.warning("CrossDocked manifest not found at %s", manifest)
        return set()
    wanted = sbdd_bench_pockets() if pockets is None else pockets
    if not wanted:
        return set()
    table = pq.read_table(manifest, columns=["complex_dir", "receptor_pdb"])
    df = table.to_pandas()
    df = df[df["complex_dir"].isin(wanted)]
    ids = df["receptor_pdb"].map(pdb_id_from_receptor).dropna()
    return set(ids)


def evaluation_pdbs(
    *,
    cd_manifest: Path | None = DEFAULT_CD_MANIFEST,
    casf_list: Path | None = DEFAULT_CASF_LIST,
    include_sbdd: bool = True,
    fold: int = 0,
) -> set[str]:
    """Every PDB id this module can name from an evaluation set, as one list.

    Each source is optional so a caller that already honours one of them (the
    atom DataModule honours the CrossDocked split directly) can leave it out
    rather than exclude the same complexes twice.

    **CASP16 is not in here, and cannot be.** The reconstruction bench indexes
    those complexes by CASP target (``L1001`` …) and ships no PDB ids, so there
    is nothing to match against a training manifest. That leaves a hole for any
    corpus drawn from a source recent enough to contain them: BioLiP2 carries
    entries up to ``9xim``, and the CASP16 ligand targets are from 2024. It is
    not a hole for the VQ-VAE, whose descriptor cache is built from CrossDocked
    alone (``--source-types cdonly``, a 2020 snapshot), which predates them --
    but that is a property of that one corpus, not something this function
    enforces. A caller reading from BioLiP has to bound the vintage itself.
    """
    out: set[str] = set()
    if cd_manifest is not None and Path(cd_manifest).exists():
        out |= crossdocked_test_pdbs(Path(cd_manifest), fold=fold)
    if casf_list is not None:
        out |= casf_pdbs(Path(casf_list))
    if include_sbdd:
        out |= SBDD_BENCH_PDBS
        # The 97-target set, by receptor PDB id. Without this the list protects
        # only the three original targets, and a PDB-id-keyed corpus (BioLiP2,
        # PLINDER) can contain 103 of the 104 receptors the generation table is
        # scored on. The CrossDocked corpus is unaffected either way -- it
        # excludes by pocket, which is the correct key there.
        if cd_manifest is not None:
            out |= sbdd_bench_receptor_pdbs(Path(cd_manifest))
    return out


@dataclass(frozen=True)
class CrossDockedSplit:
    """Which CrossDocked pocket goes to which split, and which pairs survive.

    ``pocket_split`` covers only the fold0-TRAIN pockets that were not held out
    for evaluation; a pocket absent from it is not in the corpus at all.
    ``pair_to_pocket`` is restricted the same way, so a builder can decide a
    pose's fate from its ``pair_idx`` alone.
    """

    pocket_split: dict[str, str]
    pair_to_pocket: dict[int, str]

    def split_of_pair(self, pair_idx: int) -> str | None:
        """``"train"``/``"val"`` for a pair, or ``None`` if it is excluded."""
        pocket = self.pair_to_pocket.get(int(pair_idx))
        return None if pocket is None else self.pocket_split.get(pocket)


def crossdocked_pocket_split(  # noqa: PLR0913
    manifest_path: Path,
    source_types: list[str],
    val_frac: float,
    seed: int,
    casf_ids: set[str] | None = None,
    exclude_pockets: set[str] | None = None,
) -> CrossDockedSplit:
    """The leak-free pocket-level train/val assignment for a CrossDocked corpus.

    Two corpora built from this source have to agree on composition or a
    comparison between the models trained on them measures the split as much as
    the tokenizer, so the decision lives here once rather than in each builder.

    The order of operations is load-bearing and matches the corpus that is
    already published: filter by source type, drop pairs whose receptor is a
    CASF-2016 core-set entry, keep the ``cdonly_fold0 == "train"`` pockets, drop
    the generation benchmark's evaluation pockets, then hold out ``val_frac`` of
    what remains. The permutation is drawn from ``np.random.default_rng(seed)``
    over the *sorted* pocket names, so every partition of a parallel build --
    and every rebuild, in either tokenizer -- lands on the same assignment.
    """
    import numpy as np  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    df = pq.read_table(
        manifest_path,
        columns=[
            "pair_idx",
            "complex_dir",
            "source_type",
            "cdonly_fold0",
            "receptor_pdb",
        ],
    ).to_pandas()
    df = df[df["source_type"].isin(source_types)]
    if casf_ids:
        pdb = df["receptor_pdb"].str.extract(r"^([0-9a-zA-Z]{4})_")[0].str.lower()
        df = df[~pdb.isin(casf_ids)]
    pair_to_pocket = dict(zip(df["pair_idx"], df["complex_dir"], strict=False))
    train_pockets = sorted(
        df[df["cdonly_fold0"] == "train"]["complex_dir"].dropna().unique()
    )
    if exclude_pockets:
        train_pockets = [p for p in train_pockets if p not in exclude_pockets]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_pockets))
    n_val = int(len(train_pockets) * val_frac)
    val_pockets = {train_pockets[i] for i in perm[:n_val]}
    pocket_split = {
        p: ("val" if p in val_pockets else "train") for p in train_pockets
    }
    kept = {
        int(pair): pocket
        for pair, pocket in pair_to_pocket.items()
        if pocket in pocket_split
    }
    return CrossDockedSplit(pocket_split=pocket_split, pair_to_pocket=kept)
