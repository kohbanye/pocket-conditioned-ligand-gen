"""Drive every model over a sample set and collect a long-format results table.

One row per (sample, model, modality) with reconstruction metrics. The own
model, when present, is passed in pre-materialized (it defines the sample set).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from tqdm import tqdm

from recon_bench import adapters, metrics, pb
from recon_bench.adapters.base import ReconstructionModel
from recon_bench.types import ReconResult, Sample


def _subset_to_pocket(mod, pocket: set[tuple[str, int]]):
    """Restrict a protein modality's rows to the pocket residues. Returns
    (ref, rec, n_tokens_pocket) or (None, None, None) if not applicable."""
    if mod.res_keys is None:
        return None, None, None
    keep = [i for i, k in enumerate(mod.res_keys) if k in pocket]
    if not keep:
        return None, None, None
    return mod.ref[keep], mod.rec[keep], len(keep)


_RATE_KEYS = (
    "bits_protein", "bits_ligand", "pose_bits", "arm_label", "arm_codebook", "ligand_frame",
)


def _rate_columns(mod, n_atoms: int) -> dict:
    """Cost columns, so a row is never read without its rate.

    Reconstruction error is trivially bought with a bigger codebook, so bits per
    atom belongs next to every RMSD. ``pose_bits`` is charged on top for
    tokenizers whose tokens are SE(3)-invariant and therefore need the ligand's
    placement transmitted separately.
    """
    out = {k: mod.extra.get(k) for k in _RATE_KEYS if k in mod.extra}
    bits = {
        "protein_backbone": mod.extra.get("bits_protein"),
        "protein_allatom": mod.extra.get("bits_protein"),
        "ligand": mod.extra.get("bits_ligand"),
    }.get(mod.modality)
    if mod.modality.startswith("complex"):
        # Weight the two books by how many atoms each actually encodes.
        n_prot = mod.extra.get("n_protein_rows")
        bp, bl = mod.extra.get("bits_protein"), mod.extra.get("bits_ligand")
        if None not in (n_prot, bp, bl) and n_atoms:
            bits = (n_prot * bp + (n_atoms - n_prot) * bl) / n_atoms
    if bits is not None:
        out["bits_per_atom"] = float(bits)
        # Charge the tokens actually emitted, not the atoms scored. A backbone
        # row scores 30 CA positions but the encoding still cost one token per
        # pocket atom -- billing the scored subset would flatter every model
        # whose reconstruction scope is wider than its evaluation scope.
        n_emitted = mod.n_tokens if mod.n_tokens else n_atoms
        out["total_bits"] = float(bits) * n_emitted + float(mod.extra.get("pose_bits") or 0.0)
    # An arm whose tokens are not per-atom cannot be billed as rate x tokens:
    # ESM3 spends one token per RESIDUE and ConfSeq one per SMILES symbol or
    # rotatable bond, so the two halves have different token-to-atom ratios and
    # the product above would be neither. Such an arm states its own total and
    # it wins -- the per-atom column stays for reading alongside, but the bits
    # a row is charged are the bits it actually spent.
    if mod.extra.get("total_bits") is not None:
        out["total_bits"] = float(mod.extra["total_bits"])
    return out


def _metric_row(result, mod, ref, rec, eval_scope, is_protein, *, want_pb=False) -> dict:
    # TM-score is only defined on a per-residue CA trace, so an all-atom protein
    # row gets lDDT (distance-based, atom-count agnostic) instead.
    m = metrics.all_metrics(ref, rec, protein=is_protein and mod.atom_kind == "CA")
    if is_protein and mod.atom_kind != "CA":
        m["lddt"] = metrics.lddt(ref, rec)
    if mod.modality == "ligand" and mod.extra.get("bonds"):
        m.update(metrics.bond_geometry(ref, rec, mod.extra["bonds"]))
        # End-to-end columns, scored inside the reconstruction CLI because they
        # need prolit's chemistry (see its ``end_to_end``). Absent for arms
        # whose decoder has no chemistry heads, which leaves them NaN -- read
        # as "did not take this test", not "scored zero".
        m.update(mod.extra.get("end_to_end") or {})
        if want_pb:
            m.update(
                pb.check(
                    mod.extra["elements"],
                    mod.extra["bonds"],
                    mod.extra.get("bond_orders") or [1] * len(mod.extra["bonds"]),
                    rec,
                )
            )
            # The crystal geometry through the same checker: the ceiling this
            # tokenizer could possibly reach. Whatever the reference fails is a
            # property of the data, not of the tokenizer.
            m["pb_valid_ref"] = pb.check(
                mod.extra["elements"],
                mod.extra["bonds"],
                mod.extra.get("bond_orders") or [1] * len(mod.extra["bonds"]),
                ref,
            )["pb_valid"]
    if mod.modality.startswith("complex") and "n_protein_rows" in mod.extra:
        m.update(
            metrics.complex_metrics(
                ref,
                rec,
                mod.extra["n_protein_rows"],
                mod.extra["protein_elements"],
                mod.extra["ligand_elements"],
            )
        )
    m.update(_rate_columns(mod, int(m.get("n_atoms", 0))))
    return {
        "sample_id": result.sample_id,
        "model": result.model,
        "modality": mod.modality,
        "ok": True,
        "error": None,
        "atom_kind": mod.atom_kind,
        "eval_scope": eval_scope,
        "n_residues": mod.n_residues,
        "n_tokens": mod.n_tokens,
        "runtime_s": result.runtime_s,
        **m,
    }


def _prefetch_pb(results, pb_ids: set, label: str) -> None:
    """Run this model's PoseBusters checks in parallel before scoring rows.

    Serially, PoseBusters averages ~22 s per drug-like ligand (one in our sample
    took 100 s), which is far and away the dominant cost of the benchmark. The
    checks are independent and CPU-bound, so they go to a process pool here and
    the per-row code below just reads the cache.
    """
    jobs = []
    for result in results:
        if not result.ok or result.sample_id not in pb_ids:
            continue
        for mod in result.modalities:
            if mod.modality != "ligand" or not mod.extra.get("bonds"):
                continue
            elements = mod.extra["elements"]
            bonds = mod.extra["bonds"]
            orders = mod.extra.get("bond_orders") or [1] * len(bonds)
            # Reference too: it is the ceiling each arm is judged against, and
            # the cache is content-keyed so it is only ever computed once.
            jobs.append((elements, bonds, orders, mod.rec))
            jobs.append((elements, bonds, orders, mod.ref))
    if not jobs:
        return
    n = pb.prefetch(jobs)
    print(f"[recon_bench] {label}: PoseBusters on {n} new conformers (parallel)")


def _rows_from_result(
    result: ReconResult,
    pocket_keys: dict[str, set] | None = None,
    want_pb: bool = False,
) -> list[dict]:
    if not result.ok:
        return [
            {
                "sample_id": result.sample_id,
                "model": result.model,
                "modality": None,
                "ok": False,
                "error": result.error,
                "runtime_s": result.runtime_s,
            }
        ]
    pocket = (pocket_keys or {}).get(result.sample_id)
    rows = []
    for mod in result.modalities:
        is_protein = mod.modality.startswith("protein")
        has_pocket = is_protein and pocket is not None and mod.res_keys is not None
        # Base row: the model's native reconstruction. For ESM3/FoldToken this is
        # the whole protein ("full"); the own model only ever does the pocket.
        rows.append(
            _metric_row(
                result, mod, mod.ref, mod.rec,
                "full" if has_pocket else "native", is_protein,
                want_pb=want_pb,
            )
        )
        # Extra row: the same reconstruction scored only on the pocket residues,
        # so ESM3/FoldToken (full) and the own pocket model share a row.
        if has_pocket:
            sref, srec, _ = _subset_to_pocket(mod, pocket)
            if sref is not None:
                rows.append(
                    _metric_row(result, mod, sref, srec, "pocket", is_protein)
                )
    return rows


def run(
    model_names: Iterable[str],
    samples: list[Sample],
    *,
    prebuilt: dict[str, ReconstructionModel] | None = None,
    adapter_kwargs: dict | None = None,
    pocket_keys: dict[str, set] | None = None,
    pb_valid: bool = False,
    pb_limit: int | None = None,
) -> pd.DataFrame:
    """Run each model over ``samples`` and collect metric rows.

    If ``pocket_keys`` maps sample_id -> set of (chain, resid), protein metrics
    for models that report per-residue identity are restricted to those residues
    (a full-protein reconstruction scored on the pocket residues only).

    ``pb_valid`` adds PoseBusters chemical-validity columns to ligand rows. It is
    off by default because the energy-ratio check regenerates conformers (~1 s
    per molecule, and twice that since the crystal reference is checked too);
    ``pb_limit`` caps how many samples get it, so a run can report PB on a
    subsample while every other metric covers the full set. The resulting
    ``pb_valid`` column then has its own, smaller n -- say so when reporting it.
    """
    prebuilt = prebuilt or {}
    adapter_kwargs = adapter_kwargs or {}
    rows: list[dict] = []
    if pb_valid:
        pb.quiet_rdkit()
    pb_ids = {s.sample_id for s in samples[: pb_limit or len(samples)]} if pb_valid else set()

    for name in model_names:
        model = prebuilt.get(name) or adapters.build(name, **adapter_kwargs.get(name, {}))
        try:
            model.setup()
        except Exception as exc:  # noqa: BLE001
            for s in samples:
                rows.append(
                    {"sample_id": s.sample_id, "model": name, "modality": None,
                     "ok": False, "error": f"setup failed: {exc!r}"}
                )
            continue

        applicable = [s for s in samples if model.can_protein or s.ligand_sdf]
        results = model.reconstruct_batch(tqdm(applicable, desc=name, unit="cx"))
        if pb_ids:
            _prefetch_pb(results, pb_ids, name)
        for result in results:
            rows.extend(
                _rows_from_result(
                    result, pocket_keys, want_pb=result.sample_id in pb_ids
                )
            )
        model.teardown()

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, *, fmt: bool = True) -> pd.DataFrame:
    """Per (model, modality, eval_scope) stats over successful reconstructions.

    With ``fmt=True`` (default) the metric columns are ``"mean ± std"`` strings
    for direct display; with ``fmt=False`` numeric ``*_mean`` / ``*_std`` columns
    are returned for plotting/processing.
    """
    ok = df[df["ok"]].copy()
    if ok.empty:
        return ok
    keys = ["model", "modality"]
    if "eval_scope" in ok.columns:
        ok["eval_scope"] = ok["eval_scope"].fillna("native")
        keys.append("eval_scope")
    g = ok.groupby(keys)
    metric_cols = [
        c
        for c in (
            "rmsd", "kabsch_rmsd", "tm_score", "lddt",
            # Interface: where a lost binding pose actually shows up.
            "lddt_pli", "contact_f1", "clash_lig_atom_frac", "iface_lig_rmsd",
            # Chemical validity of the reconstruction (opt-in; smaller n).
            "pb_valid", "pb_valid_ref",
            # Ligand-internal geometry, mean and worst-case per molecule.
            "bond_mae", "bond_max", "angle_mae", "angle_max",
            # End-to-end: decoded chemistry only, no reference graph handed back.
            "chem_element", "chem_charge", "chem_numH", "chem_determining",
            "graph_exact", "graph_missing", "graph_extra",
            "mol_buildable", "smiles_match", "smiles_match_true_graph",
            "smiles_ref_buildable",
        )
        if c in ok.columns
    ]

    out: dict = {"n": g.size()}
    for c in metric_cols:
        mean, std = g[c].mean(), g[c].std()
        if fmt:
            out[c] = [
                "—" if pd.isna(m) else f"{m:.2f} ± {0.0 if pd.isna(s) else s:.2f}"
                for m, s in zip(mean, std, strict=False)
            ]
        else:
            out[f"{c}_mean"] = mean.round(4)
            out[f"{c}_std"] = std.round(4)
    for c in ("n_tokens", "n_atoms", "bits_per_atom", "pose_bits", "total_bits"):
        if c in ok.columns:
            out[c] = g[c].mean().round(2)
    return pd.DataFrame(out).reset_index()
