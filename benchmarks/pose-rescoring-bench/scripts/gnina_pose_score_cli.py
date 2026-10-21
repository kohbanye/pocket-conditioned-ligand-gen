"""Score the CASF-2016 docking-power decoy stacks with gnina's CNN rescorer.

Driven by ``pose_rescoring_bench.adapters.gnina``. Unlike the two DGL baselines
this one is not a Python library: gnina is an executable, so this script only
marshals inputs, runs it once per target, and parses what it prints.

gnina is run in ``--score_only`` mode, which scores the poses exactly as given
without moving them -- the same contract as every other row in this table.
``--cnn_scoring rescore`` runs the CNN on the final pose only, which is what
``--score_only`` means for the CNN and is gnina's documented rescoring mode.

Three numbers come back per pose. ``CNNscore`` is the pose classifier -- the
probability the CNN assigns to this being a good pose -- and it is the one this
bench reports as ``head``, because docking power asks exactly that question.
``CNNaffinity`` and the empirical ``Affinity`` are written to a side file rather
than thrown away: the first is what an affinity table would want, and the second
is gnina's built-in Vina-like term, which is a useful check that the run was
sane. Reporting the best of the three would be selection on the test set.

Pose identity comes from the order gnina prints in, which is the order of the
models in the sdf, cross-checked against ``## Name`` lines when gnina emits
them. The count is asserted against the sdf, so a silently truncated run fails
here rather than becoming a target scored on half its decoys.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

#: gnina prints one of these per scored pose.
_AFFINITY = re.compile(r"^Affinity:\s+(-?[\d.]+)", re.MULTILINE)
_CNNSCORE = re.compile(r"^CNNscore:\s+(-?[\d.]+)", re.MULTILINE)
_CNNAFFINITY = re.compile(r"^CNNaffinity:\s+(-?[\d.]+)", re.MULTILINE)
#: Emitted for named ligands; absent for some writers, hence the order fallback.
_NAME = re.compile(r"^##\s+Name\s+(\S+)", re.MULTILINE)


def sdf_pose_names(sdf: Path) -> list[str]:
    """Title-line names of every model in a multi-model sdf, in file order."""
    names, expect_title = [], True
    for line in sdf.read_text(errors="replace").splitlines():
        if expect_title:
            names.append(line.strip())
            expect_title = False
        elif line.startswith("$$$$"):
            expect_title = True
    # A trailing "$$$$" arms the flag once more than there are molecules.
    if names and names[-1] == "":
        names.pop()
    return names


def decoys_sdf(pdbid: str, decoys_mol2: Path, sdf_dir: Path, obabel: str) -> Path:
    """Convert a decoy stack to sdf once, then reuse it.

    Shares the layout of the DGL driver's cache so a second baseline does not
    redo the conversion, but writes into its own directory: the two are run by
    different jobs and a half-written file read by the other is not worth the
    saved minute.
    """
    out = sdf_dir / f"{pdbid}_decoys.sdf"
    if out.is_file() and out.stat().st_size > 0:
        return out
    subprocess.run(  # noqa: S603
        [obabel, str(decoys_mol2), "-O", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def score_target(
    gnina_cmd: list[str],
    receptor: Path,
    ligands: Path,
    cnn_scoring: str,
    seed: int,
) -> tuple[list[float], list[float], list[float], list[str]]:
    """Run gnina once and return (cnnscore, cnnaffinity, affinity, names)."""
    cmd = [
        *gnina_cmd,
        "-r", str(receptor),
        "-l", str(ligands),
        "--score_only",
        "--cnn_scoring", cnn_scoring,
        "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        tail = (proc.stdout[-2000:] + proc.stderr[-2000:]).strip()
        msg = f"gnina failed (rc={proc.returncode}): {tail}"
        raise RuntimeError(msg)
    out = proc.stdout
    return (
        [float(x) for x in _CNNSCORE.findall(out)],
        [float(x) for x in _CNNAFFINITY.findall(out)],
        [float(x) for x in _AFFINITY.findall(out)],
        _NAME.findall(out),
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gnina", required=True,
                    help="the gnina command, shell-quoted (may be a wrapper)")
    ap.add_argument("--casf", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cnn-scoring", default="rescore",
                    choices=("none", "rescore", "refinement", "all"))
    ap.add_argument("--obabel", default="obabel")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pdbids", default=None, help="comma-separated subset")
    return ap.parse_args()


def load_targets(casf: Path) -> list[str]:
    """The docking-power target list (the targets that ship a decoy stack)."""
    import pandas as pd  # noqa: PLC0415

    core = pd.read_csv(
        casf / "power_docking" / "CoreSet.dat", sep=r"[,\t ]+", engine="python"
    )
    return [str(x) for x in core["#code"]]


def main() -> None:
    args = _parse_args()

    import shlex  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    gnina_cmd = shlex.split(args.gnina)
    coreset = args.casf / "coreset"
    decoys_dir = args.casf / "decoys_docking"
    sdf_dir = args.work / "decoys_sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    targets = args.pdbids.split(",") if args.pdbids else load_targets(args.casf)
    print(f"gnina cnn_scoring={args.cnn_scoring} targets={len(targets)}", flush=True)

    rows, extra, failed = [], [], []
    for n, pdbid in enumerate(targets, 1):
        try:
            sdf = decoys_sdf(
                pdbid, decoys_dir / f"{pdbid}_decoys.mol2", sdf_dir, args.obabel
            )
            names = sdf_pose_names(sdf)
            cnns, cnnaff, aff, printed = score_target(
                gnina_cmd,
                coreset / pdbid / f"{pdbid}_protein.pdb",
                sdf,
                args.cnn_scoring,
                args.seed,
            )
            # All three streams are read by position, so a mismatch in any of
            # them would pair a score with the wrong pose rather than fail.
            got = {"CNNscore": len(cnns), "CNNaffinity": len(cnnaff),
                   "Affinity": len(aff)}
            if set(got.values()) != {len(names)}:
                msg = (
                    f"gnina emitted {got} for an sdf holding {len(names)} "
                    "poses; refusing a target whose scores cannot be matched "
                    "to poses by position"
                )
                raise ValueError(msg)  # noqa: TRY301
            # Prefer the names gnina echoed; fall back to file order.
            pose_names = printed if len(printed) == len(names) else names
            df_score = pd.DataFrame({"#code": pose_names, "score": cnns})
            df_rmsd = pd.read_csv(
                decoys_dir / f"{pdbid}_rmsd.dat", sep=r"[, \t]+",
                header=0, engine="python",
            )
            merged = df_rmsd.merge(df_score, on="#code")
            merged = merged[~merged["#code"].str.contains("_ligand")]
            if merged.empty:
                msg = "no poses survived the rmsd join"
                raise ValueError(msg)  # noqa: TRY301
        except Exception as exc:  # noqa: BLE001
            failed.append((pdbid, repr(exc)[:160]))
            print(f"[{n}/{len(targets)}] {pdbid} FAILED: {exc!r:.160}", flush=True)
            continue

        for rec in merged[["#code", "rmsd", "score"]].to_dict("records"):
            rows.append(  # noqa: PERF401
                {"pdbid": pdbid, "pose": rec["#code"],
                 "rmsd": float(rec["rmsd"]), "head": float(rec["score"])}
            )
        # Lengths were checked above, so positional pairing is sound here.
        for name, ca, a in zip(pose_names, cnnaff, aff):  # noqa: B905
            extra.append(
                {"pdbid": pdbid, "pose": name, "cnnaffinity": ca, "affinity": a}
            )
        if n % 25 == 0:
            print(f"[{n}/{len(targets)}] {pdbid} n_poses={len(merged)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pdbid", "pose", "rmsd", "head"])
        writer.writeheader()
        writer.writerows(rows)
    side = args.out.with_name(args.out.stem + "_extra.csv")
    with side.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["pdbid", "pose", "cnnaffinity", "affinity"]
        )
        writer.writeheader()
        writer.writerows(extra)
    scored = len({r["pdbid"] for r in rows})
    print(
        f"scored {scored}/{len(targets)} targets, {len(rows)} poses -> {args.out}",
        flush=True,
    )
    if failed:
        print(f"FAILED {len(failed)}: {failed[:10]}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
