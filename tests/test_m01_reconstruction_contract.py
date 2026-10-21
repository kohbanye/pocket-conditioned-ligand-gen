"""M01 gate derives its frame and rejects hidden cohort or label differences."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs/research/affinity_fresh_20260913/audit_m01_reconstruction.py"
)
if not SCRIPT.exists():
    # docs/ is untracked -- it is the investigation record, written beside the
    # weights on the machine that has them. Skip at module level rather than
    # letting exec_module raise, because an import error during collection
    # takes the whole suite down instead of this one file.
    pytest.skip(
        "docs/ is not tracked; the audited script is only on the machine that "
        "produced the run",
        allow_module_level=True,
    )

SPEC = importlib.util.spec_from_file_location("m01_audit_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_args(tmp_path: Path) -> SimpleNamespace:
    row = {
        "pdbid": "1abc",
        "ccd": "LIG",
        "serial": "1",
        "receptor_chain": "A",
        "ligand_chain": "A",
        "status": "accepted",
        "split": "val",
        "group": 5,
        "pk": 7.2,
    }
    for name, schema in [("trial", "validated_v1"), ("control", "legacy")]:
        directory = tmp_path / name
        directory.mkdir()
        torch.save(
            {
                "pocket_distance": "heavy",
                "max_residues": 37,
                "chemistry_schema": schema,
                "validated_cohort": True,
            },
            directory / "meta.pt",
        )
        (directory / "complex_manifest.jsonl").write_text(json.dumps(row) + "\n")
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("2def\n")
    return SimpleNamespace(
        synthetic_smoke=False,
        site_manifest=tmp_path / "trial/complex_manifest.jsonl",
        control_manifest=tmp_path / "control/complex_manifest.jsonl",
        exclude_pdbs=excluded,
    )


def test_crop_is_derived_from_matching_metadata(tmp_path: Path) -> None:
    args = fixture_args(tmp_path)
    cfg, rows, coverage = MODULE.development_contract(args)
    assert cfg.pocket_distance == "heavy"
    assert cfg.max_residues == 37
    assert len(rows) == 1
    assert coverage["status"] == "exact_accepted_validation_match"


@pytest.mark.parametrize(
    ("field", "value"),
    [("pk", 6.1), ("group", 9), ("split", "train"), ("pdbid", "3ghi")],
)
def test_coverage_and_labels_must_match(
    tmp_path: Path, field: str, value: object
) -> None:
    args = fixture_args(tmp_path)
    row = json.loads(args.control_manifest.read_text())
    row[field] = value
    args.control_manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=r"mismatch|coverage differs"):
        MODULE.development_contract(args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pocket_distance", "ca"),
        ("max_residues", 50),
        ("chemistry_schema", "validated_v1"),
    ],
)
def test_control_metadata_cannot_silently_change_recipe(
    tmp_path: Path, field: str, value: object
) -> None:
    args = fixture_args(tmp_path)
    path = args.control_manifest.parent / "meta.pt"
    meta = torch.load(path, weights_only=False)
    meta[field] = value
    torch.save(meta, path)
    with pytest.raises(ValueError, match=r"crop|max_residues|chemistry"):
        MODULE.development_contract(args)


def test_synthetic_default_is_explicit() -> None:
    cfg, rows, coverage = MODULE.development_contract(
        SimpleNamespace(synthetic_smoke=True)
    )
    assert cfg.pocket_distance == "ca"
    assert cfg.max_residues == 50
    assert rows == []
    assert coverage["status"] == "synthetic_smoke"
