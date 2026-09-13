"""Assemble the tables from the ``results/`` dump tree.

Layout::

    results/<arm>/<head-label>.csv      our arms, one CSV per head
    results/{genscore,boltz2,vina}/scoring.csv   baselines

An arm whose dumps are absent is skipped rather than erroring, so the table
grows as the improvement loop trains heads, and a partially-run night still
produces a readable comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from affinity_bench import aggregate
from affinity_bench import metrics as A
from affinity_bench.variants import (
    ARM_ORDER,
    SEED_FAMILIES,
    SEED_REFERENCE,
    SEED_REFERENCE_FOR,
)

if TYPE_CHECKING:
    from pathlib import Path

Report = tuple[pd.DataFrame, pd.DataFrame]  # (metrics, significance)

#: Baselines: the name tables show, the directory, and the column that dump
#: calls its prediction. ``collect_baselines.py`` writes ``head``; the legacy
#: names are accepted so dumps carried over from the source repo read unchanged.
BASELINES = (
    ("GenScore", "genscore", "score"),
    ("Boltz-2", "boltz2", "score"),
    ("Vina", "vina", "vina_score"),
)
#: Every method this bench is judged against beats-or-not.
REFERENCE = "GenScore"


def _arm_pred(arm_dir: Path) -> pd.DataFrame | None:
    """One arm's prediction frame: its single head, or a z-sum of its heads."""
    members = sorted(arm_dir.glob("*.csv"))
    if not members:
        return None
    frames = [pd.read_csv(m) for m in members]
    return frames[0] if len(frames) == 1 else A.zsum_ensemble(frames)


#: Directories under ``results/`` that are not arms.
_NOT_ARMS = frozenset(
    {"tables", "figures", "archive", *(sub for _, sub, _ in BASELINES)},
)


def discover_arms(results_dir: Path, protocol: str = "_f16") -> list[str]:
    """Every arm directory holding dumps for ONE protocol, registered ones first.

    Discovered rather than listed, so a head trained overnight appears in the
    table as soon as its dump lands, without an edit here. Registered arms lead
    because they are the ones the paper names; the rest follow alphabetically.

    ``protocol`` filters by the directory suffix, and it is not cosmetic: dumps
    for other evaluation sets live here too (``_pbhold`` is the held-out PDBbind
    slice, 281 complexes with no CASF clusters). Pooling them into one table
    puts an R computed on 281 complexes of a different, harder set next to one
    computed on the CASF 285, in the same column, as though they compared --
    and ranking rho reads NaN there because every complex is its own singleton.
    """
    if not results_dir.is_dir():
        return []
    found = {
        d.name
        for d in results_dir.iterdir()
        if d.is_dir()
        and d.name not in _NOT_ARMS
        and d.name.endswith(protocol)
        and any(d.glob("*.csv"))
    }
    ordered = [a for a in ARM_ORDER if a in found]
    return ordered + sorted(found - set(ordered))


def methods(
    results_dir: Path,
    arms: tuple[str, ...] | list[str] | None = None,
    protocol: str = "_f16",
) -> dict[str, pd.DataFrame]:
    """Our arms plus every baseline present, keyed by the name tables show.

    One protocol at a time: the baselines are CASF-only, so mixing evaluation
    sets would compare them against numbers from a set they were never run on.
    """
    out: dict[str, pd.DataFrame] = {}
    for name in arms if arms is not None else discover_arms(results_dir, protocol):
        pred = _arm_pred(results_dir / name)
        if pred is not None:
            out[f"OURS ({name})"] = pred
    for label, sub, legacy_col in BASELINES:
        path = results_dir / sub / "scoring.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "head" not in df.columns:
            if legacy_col not in df.columns:
                msg = (
                    f"{path}: no prediction column; expected 'head' or "
                    f"{legacy_col!r}, have {list(df.columns)}"
                )
                raise ValueError(msg)
            df = df.rename(columns={legacy_col: "head"})
        out[label] = df
    return out


def comparison(results_dir: Path) -> Report:
    """Every method against the reference, with significance."""
    preds = methods(results_dir)
    if not preds:
        return pd.DataFrame(), pd.DataFrame()
    metrics = aggregate.affinity_metrics(preds)
    ref = REFERENCE if REFERENCE in preds else next(iter(preds))
    sig = (
        aggregate.affinity_pairwise(preds, reference=ref)
        if len(preds) > 1
        else pd.DataFrame()
    )
    return metrics, sig


def scoreboard(results_dir: Path) -> pd.DataFrame:
    """The goal restated per arm: does it beat the reference on BOTH metrics?

    Printed next to the table so a night's work is read against the target that
    was set, not against whichever metric moved.
    """
    preds = methods(results_dir)
    ours = [m for m in preds if m.startswith("OURS")]
    if REFERENCE not in preds or not ours:
        return pd.DataFrame()
    return pd.DataFrame(
        {m: aggregate.beats_reference(preds, m, REFERENCE) for m in ours},
    ).T


def seed_families(
    results_dir: Path,
    protocol: str = "_f16",
) -> dict[str, dict[int, pd.DataFrame]]:
    """Load every registered seed family's dumps: family -> seed -> frame.

    A family whose seeds are not all present is still returned with the ones
    that are, so a partially-finished sweep reads as "2 of 3 seeds" rather than
    disappearing.
    """
    out: dict[str, dict[int, pd.DataFrame]] = {}
    for family, by_seed in SEED_FAMILIES.items():
        frames: dict[int, pd.DataFrame] = {}
        for seed, arm in by_seed.items():
            pred = _arm_pred(results_dir / f"{arm}{protocol}")
            if pred is not None:
                frames[seed] = pred
        if frames:
            out[family] = frames
    return out


def seed_report(results_dir: Path, protocol: str = "_f16") -> Report:
    """Per-family mean +/- sd over seeds, and the paired comparison between them.

    This is the table to read, not the per-arm one: a row there is one draw from
    a distribution whose sd is larger than the differences being argued about.
    """
    families = seed_families(results_dir, protocol)
    if not families:
        return pd.DataFrame(), pd.DataFrame()
    spread = pd.DataFrame(
        {name: aggregate.seed_spread(frames) for name, frames in families.items()},
    ).T
    # One paired table per control, because a family is only comparable to an
    # arm that differs from it by one thing; see variants.SEED_REFERENCE_FOR.
    # Looked up from THIS module's names, not through a helper in variants:
    # a helper would close over variants' own globals, so overriding
    # SEED_REFERENCE or SEED_REFERENCE_FOR here would silently not apply and the
    # names imported above would be decorative. That bug shipped once.
    blocks = []
    by_ref: dict[str, list[str]] = {}
    for name in families:
        by_ref.setdefault(SEED_REFERENCE_FOR.get(name, SEED_REFERENCE), []).append(name)
    for ref, names in by_ref.items():
        if ref not in families:
            continue
        subset = {ref: families[ref], **{n: families[n] for n in names if n != ref}}
        if len(subset) < 2:  # noqa: PLR2004
            continue
        blocks.append(aggregate.paired_by_seed(subset, reference=ref))
    paired = pd.concat(blocks) if blocks else pd.DataFrame()
    return spread, paired
