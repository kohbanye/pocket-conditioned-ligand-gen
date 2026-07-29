"""Build a HuggingFace Hub dataset from local CrossDocked2020 data.

Parses types files, creates receptor archives, ligand tar shards,
and a Parquet manifest that replaces the ~43GB types files.

Usage:
    uv run python pipelines/corpora/build_hf_dataset.py [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import tarfile
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types file parsing
# ---------------------------------------------------------------------------

_RE_GNINATYPES_IDX = re.compile(r"_\d+\.gninatypes$")


def _gninatypes_to_pdb(path: str) -> str:
    return _RE_GNINATYPES_IDX.sub(".pdb", path)


def _gninatypes_to_sdf(path: str) -> str:
    return _RE_GNINATYPES_IDX.sub(".sdf.gz", path)


def _detect_source_type(filename: str) -> str:
    """Detect the types file category from its filename."""
    name = Path(filename).stem
    if name.startswith("cdonly"):
        return "cdonly"
    if "redocked" in name:
        return "it2_redocked"
    if name.startswith("it0"):
        return "it0"
    return "other"


def _detect_fold(filename: str) -> int | None:
    """Extract fold number (0, 1, 2) from types filename."""
    name = Path(filename).stem
    for fold in (0, 1, 2):
        if f"train{fold}" in name or f"test{fold}" in name:
            return fold
    return None


def _detect_split(filename: str) -> str | None:
    """Detect train/test from types filename."""
    name = Path(filename).stem
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    return None


def _parse_types_file_streaming(
    types_path: Path,
) -> dict[tuple[str, str], dict]:
    """Parse a types file line-by-line, returning deduplicated pair metadata.

    Returns a dict mapping (receptor_pdb, ligand_sdf) -> {label, score1, score2}.
    Only keeps the first occurrence's metadata for each unique pair.
    """
    source_type = _detect_source_type(types_path.name)
    fold = _detect_fold(types_path.name)
    split = _detect_split(types_path.name)

    pairs: dict[tuple[str, str], dict] = {}

    with types_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:  # noqa: PLR2004
                continue

            rec_pdb = _gninatypes_to_pdb(parts[3])
            lig_sdf = _gninatypes_to_sdf(parts[4])
            key = (rec_pdb, lig_sdf)

            if key not in pairs:
                try:
                    label = int(parts[0])
                    score1 = float(parts[1])
                    score2 = float(parts[2])
                except ValueError:
                    label, score1, score2 = 0, 0.0, 0.0

                pairs[key] = {
                    "label": label,
                    "score1": score1,
                    "score2": score2,
                    "source_type": source_type,
                }

    logger.info(
        "  %s: %d unique pairs (source=%s, fold=%s, split=%s)",
        types_path.name,
        len(pairs),
        source_type,
        fold,
        split,
    )
    return pairs


def _parse_all_types_files(
    types_dir: Path,
) -> tuple[list[dict], dict[tuple[str, str], dict[str, str | None]]]:
    """Parse all types files and build a unified pair list.

    Returns:
        pairs: sorted list of dicts with pair info
        fold_splits: mapping (rec, lig) -> {
            "cdonly_fold0": "train"/"test"/None, ...
        }
    """
    # Collect all types files
    types_files = sorted(types_dir.glob("*.types"))
    if not types_files:
        msg = f"No .types files found in {types_dir}"
        raise FileNotFoundError(msg)

    logger.info("Found %d types files in %s", len(types_files), types_dir)

    # Global pair registry: (rec, lig) -> metadata from first occurrence
    global_pairs: dict[tuple[str, str], dict] = {}
    # Track fold splits per pair per category
    fold_splits: dict[tuple[str, str], dict[str, str | None]] = {}

    for types_file in types_files:
        source_type = _detect_source_type(types_file.name)
        fold = _detect_fold(types_file.name)
        split = _detect_split(types_file.name)

        file_pairs = _parse_types_file_streaming(types_file)

        for key, meta in file_pairs.items():
            # Register pair if new
            if key not in global_pairs:
                global_pairs[key] = meta
                fold_splits[key] = {}

            # Record fold/split info
            if fold is not None and split is not None:
                fold_key = f"{source_type}_fold{fold}"
                fold_splits[key][fold_key] = split

    logger.info("Total unique pairs across all types files: %d", len(global_pairs))

    # Sort pairs deterministically
    sorted_keys = sorted(global_pairs.keys())
    pairs = []
    for idx, key in enumerate(sorted_keys):
        rec_pdb, lig_sdf = key
        meta = global_pairs[key]
        complex_dir = str(Path(rec_pdb).parent)
        pairs.append(
            {
                "pair_idx": idx,
                "complex_dir": complex_dir,
                "receptor_pdb": Path(rec_pdb).name,
                "ligand_sdf_gz": Path(lig_sdf).name,
                "source_type": meta["source_type"],
                "label": meta["label"],
                "score1": meta["score1"],
                "score2": meta["score2"],
            }
        )

    return pairs, {k: fold_splits[k] for k in sorted_keys}


# ---------------------------------------------------------------------------
# Receptor archive creation
# ---------------------------------------------------------------------------


def _build_receptor_archives(
    pairs: list[dict],
    crossdocked_dir: Path,
    output_dir: Path,
    max_per_shard: int = 50000,
) -> int:
    """Create tar.gz archives of unique receptor PDB files.

    Returns the number of unique receptors archived.
    """
    receptors_dir = output_dir / "receptors"
    receptors_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique receptor paths
    unique_receptors: set[str] = set()
    for pair in pairs:
        rec_path = f"{pair['complex_dir']}/{pair['receptor_pdb']}"
        unique_receptors.add(rec_path)

    sorted_receptors = sorted(unique_receptors)
    logger.info("Archiving %d unique receptor PDB files", len(sorted_receptors))

    shard_idx = 0
    archived = 0
    tar_path = receptors_dir / f"shard-{shard_idx:03d}.tar.gz"
    tar: tarfile.TarFile | None = None

    try:
        tar = tarfile.open(tar_path, "w:gz")  # noqa: SIM115
        for i, rec_rel in enumerate(tqdm(sorted_receptors, desc="Receptors")):
            src = crossdocked_dir / rec_rel
            if not src.exists():
                logger.warning("Receptor not found: %s", src)
                continue
            tar.add(str(src), arcname=rec_rel)
            archived += 1

            if (i + 1) % max_per_shard == 0 and (i + 1) < len(sorted_receptors):
                tar.close()
                shard_idx += 1
                tar_path = receptors_dir / f"shard-{shard_idx:03d}.tar.gz"
                tar = tarfile.open(tar_path, "w:gz")  # noqa: SIM115
    finally:
        if tar is not None:
            tar.close()

    logger.info(
        "Created %d receptor archive(s), %d files total",
        shard_idx + 1,
        archived,
    )
    return archived


# ---------------------------------------------------------------------------
# Ligand tar shard creation
# ---------------------------------------------------------------------------


def _build_ligand_shards(
    pairs: list[dict],
    crossdocked_dir: Path,
    output_dir: Path,
    target_shard_bytes: int = 500 * 1024 * 1024,
) -> list[int]:
    """Create tar shards of ligand SDF.gz files with JSON metadata.

    Returns list of shard_idx for each pair (aligned with pairs list).
    """
    ligands_dir = output_dir / "ligands"
    ligands_dir.mkdir(parents=True, exist_ok=True)

    shard_indices: list[int] = []
    shard_idx = 0
    current_shard_bytes = 0
    tar: tarfile.TarFile | None = None

    tar_path = ligands_dir / f"{shard_idx:06d}.tar"

    try:
        tar = tarfile.open(tar_path, "w")  # noqa: SIM115
        for pair in tqdm(pairs, desc="Ligands"):
            pair_idx = pair["pair_idx"]
            lig_rel = f"{pair['complex_dir']}/{pair['ligand_sdf_gz']}"
            src = crossdocked_dir / lig_rel

            # SDF.gz file
            if src.exists():
                sdf_data = src.read_bytes()
            else:
                logger.warning("Ligand not found: %s", src)
                sdf_data = b""

            sdf_name = f"{pair_idx:07d}.sdf.gz"
            info = tarfile.TarInfo(name=sdf_name)
            info.size = len(sdf_data)
            tar.addfile(info, BytesIO(sdf_data))
            current_shard_bytes += len(sdf_data)

            # JSON metadata
            meta = {
                "pair_idx": pair_idx,
                "receptor_path": (f"{pair['complex_dir']}/{pair['receptor_pdb']}"),
                "complex_dir": pair["complex_dir"],
                "ligand_original_name": pair["ligand_sdf_gz"],
                "source_type": pair["source_type"],
            }
            meta_bytes = json.dumps(meta, ensure_ascii=False).encode()
            json_name = f"{pair_idx:07d}.json"
            info = tarfile.TarInfo(name=json_name)
            info.size = len(meta_bytes)
            tar.addfile(info, BytesIO(meta_bytes))
            current_shard_bytes += len(meta_bytes)

            shard_indices.append(shard_idx)

            # Start new shard if current one exceeds target size
            if current_shard_bytes >= target_shard_bytes:
                tar.close()
                shard_idx += 1
                tar_path = ligands_dir / f"{shard_idx:06d}.tar"
                tar = tarfile.open(tar_path, "w")  # noqa: SIM115
                current_shard_bytes = 0
    finally:
        if tar is not None:
            tar.close()

    logger.info("Created %d ligand shard(s)", shard_idx + 1)
    return shard_indices


# ---------------------------------------------------------------------------
# Manifest creation
# ---------------------------------------------------------------------------

# All possible fold column names across categories and folds
_FOLD_CATEGORIES = ("cdonly", "it0", "it2_redocked")
_FOLD_NUMS = (0, 1, 2)
_FOLD_COLUMNS = [f"{cat}_fold{n}" for cat in _FOLD_CATEGORIES for n in _FOLD_NUMS]


def _build_manifest(
    pairs: list[dict],
    fold_splits: dict[tuple[str, str], dict[str, str | None]],
    shard_indices: list[int],
    output_dir: Path,
) -> None:
    """Write manifest.parquet with pair index and metadata."""
    # Reconstruct sorted keys for fold_splits lookup
    sorted_keys = [
        (
            f"{p['complex_dir']}/{p['receptor_pdb']}",
            f"{p['complex_dir']}/{p['ligand_sdf_gz']}",
        )
        for p in pairs
    ]

    columns = {
        "pair_idx": pa.array([p["pair_idx"] for p in pairs], type=pa.uint32()),
        "complex_dir": pa.array([p["complex_dir"] for p in pairs], type=pa.string()),
        "receptor_pdb": pa.array([p["receptor_pdb"] for p in pairs], type=pa.string()),
        "ligand_sdf_gz": pa.array(
            [p["ligand_sdf_gz"] for p in pairs], type=pa.string()
        ),
        "source_type": pa.array([p["source_type"] for p in pairs], type=pa.string()),
        "shard_idx": pa.array(shard_indices, type=pa.uint16()),
        "label": pa.array([p["label"] for p in pairs], type=pa.int8()),
        "score1": pa.array([p["score1"] for p in pairs], type=pa.float32()),
        "score2": pa.array([p["score2"] for p in pairs], type=pa.float32()),
    }

    # Add fold split columns
    for col_name in _FOLD_COLUMNS:
        values = []
        for key in sorted_keys:
            splits = fold_splits.get(key, {})
            values.append(splits.get(col_name))
        columns[col_name] = pa.array(values, type=pa.string())

    table = pa.table(columns)
    manifest_path = output_dir / "manifest.parquet"
    pq.write_table(table, manifest_path, compression="zstd")

    logger.info(
        "Wrote manifest: %d rows, %d columns, %.1f MB",
        table.num_rows,
        table.num_columns,
        manifest_path.stat().st_size / 1e6,
    )


# ---------------------------------------------------------------------------
# README generation
# ---------------------------------------------------------------------------


def _write_readme(output_dir: Path, num_pairs: int, num_receptors: int) -> None:
    readme = output_dir / "README.md"
    readme.write_text(f"""\
---
license: cc0-1.0
task_categories:
  - other
tags:
  - drug-discovery
  - molecular-generation
  - protein-ligand
  - structural-biology
size_categories:
  - 1M<n<10M
---

# CrossDocked2020

Pre-processed CrossDocked2020 dataset containing raw receptor PDB and ligand
SDF.gz files, organized for efficient loading.

## Dataset Summary

- **Unique pairs**: {num_pairs:,}
- **Unique receptor PDB files**: {num_receptors:,}
- **Source types**: cdonly, it0, it2_redocked
- **Fold splits**: 3 folds (0, 1, 2) per source type category

## Repository Structure

```
receptors/          Unique receptor PDB files in tar.gz archives
ligands/            Ligand SDF.gz files in tar shards (WebDataset-compatible)
manifest.parquet    Pair index with metadata and fold split info
```

## Ligand Tar Shard Format

Each shard is a tar file containing pairs of files per sample:
- `{{pair_idx:07d}}.sdf.gz` — original ligand SDF.gz (all conformers)
- `{{pair_idx:07d}}.json` — metadata (receptor_path, complex_dir, source_type)

## Manifest Schema

| Column | Type | Description |
|--------|------|-------------|
| pair_idx | uint32 | Global unique pair ID |
| complex_dir | string | Complex directory name |
| receptor_pdb | string | Receptor PDB filename |
| ligand_sdf_gz | string | Ligand SDF.gz filename |
| source_type | string | cdonly / it0 / it2_redocked |
| shard_idx | uint16 | Ligand shard number |
| label | int8 | Types file label (0/1) |
| score1 | float32 | Types file score 1 |
| score2 | float32 | Types file score 2 |
| {{cat}}_fold{{n}} | string | "train" / "test" per category and fold |

## Original Source

Paul G. Francoeur, Tomohide Masuda, Jocelyn Sunseri, Andrew Jia,
Richard B. Iovanisci, Ian Snyder, David R. Koes.
*J. Chem. Inf. Model.* 2020, 60(9), p.4200-4215.
""")
    logger.info("Wrote %s", readme)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory containing CrossDocked2020/ and types/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hf_dataset"),
        help="Output directory for the HuggingFace dataset",
    )
    parser.add_argument(
        "--shard-size-mb",
        type=int,
        default=500,
        help="Target size per ligand tar shard in MB",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir
    crossdocked_dir = data_dir / "CrossDocked2020"
    types_dir = data_dir / "types"

    if not crossdocked_dir.exists():
        msg = f"CrossDocked2020 directory not found: {crossdocked_dir}"
        raise FileNotFoundError(msg)
    if not types_dir.exists():
        msg = f"Types directory not found: {types_dir}"
        raise FileNotFoundError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse all types files
    logger.info("=== Step 1: Parsing types files ===")
    pairs, fold_splits = _parse_all_types_files(types_dir)

    # Step 2: Build receptor archives
    logger.info("=== Step 2: Building receptor archives ===")
    num_receptors = _build_receptor_archives(pairs, crossdocked_dir, output_dir)

    # Step 3: Build ligand tar shards
    logger.info("=== Step 3: Building ligand tar shards ===")
    shard_indices = _build_ligand_shards(
        pairs,
        crossdocked_dir,
        output_dir,
        target_shard_bytes=args.shard_size_mb * 1024 * 1024,
    )

    # Step 4: Write manifest
    logger.info("=== Step 4: Writing manifest ===")
    _build_manifest(pairs, fold_splits, shard_indices, output_dir)

    # Step 5: Write README
    logger.info("=== Step 5: Writing README ===")
    _write_readme(output_dir, len(pairs), num_receptors)

    logger.info("Done! Dataset written to %s", output_dir)


if __name__ == "__main__":
    main()
