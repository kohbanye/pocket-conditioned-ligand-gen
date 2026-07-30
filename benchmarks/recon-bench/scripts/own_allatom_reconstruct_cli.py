"""Reconstruct (receptor, ligand) pairs with the own **all-atom** tokenizer.

Counterpart of ``own_reconstruct_cli.py`` for the all-atom tokenizer family that
replaced the residue-level one. Run by the **own model's** interpreter (it
imports that repo's code), not the bench env.

Unlike the residue-level CLI, this one dumps NPZ rather than PDB: the bench needs
per-atom correspondence, ligand bond orders, and token counts, all of which a PDB
round-trip would lose or blur.

One "arm" = one tokenizer configuration, given as JSON:

    {"kind": "vq",                       # or "binning" (no weights)
     "protein_ckpt": ..., "protein_norm": ...,
     "ligand_ckpt": ...,  "ligand_norm": ...,
     "ligand_frame": "pocket" | "local",  # frame the ligand is encoded in
     "pose_bits": null | 13 | 26 | ...}   # budget to transmit the ligand pose

``ligand_frame="local"`` mimics a single-modality ligand tokenizer (Mol-StrucTok,
Geo2Seq): the molecule is encoded in its own canonical frame, so the tokens are
SE(3)-invariant and carry no pose at all. Placing it back into the receptor then
costs a rigid transform, which ``pose_bits`` quantizes -- ``null`` means oracle
placement (the unattainable upper bound).

Invoked as::

    <own_venv_python> own_allatom_reconstruct_cli.py \
        --workdir <own_repo> --arm arm.json --pairs pairs.json --out-dir <out>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def load_own(workdir: Path):
    """Put the own repo on the path and import the pieces this CLI needs."""
    if str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))
    from prolit.config import PocketExtractionConfig
    from prolit.model.vqvae_module import AtomVQVAEModule
    from prolit.tokenizers.atom import (
        LigandAtomDescriptor,
        ProteinAtomDescriptor,
        atom_descriptor_to_coords,
        precompute_receptor_atom_features,
    )
    from prolit.tokenizers.descriptor_schema import (
        ATOM_DESCRIPTOR_DIM,
        ATOM_LAYOUT,
        LIGAND_ELEMENT_VOCAB,
        fields_by_name,
    )
    from prolit.tokenizers.geometry import (
        cartesian_to_spherical_np,
        spherical_to_cartesian_np,
    )
    from prolit.tokenizers.ligand import parse_sdf
    from prolit.tokenizers.protein import (
        compute_canonical_frame,
        extract_pocket_atoms_from_candidates,
        precompute_pocket_atom_candidates,
    )

    return dict(
        PocketExtractionConfig=PocketExtractionConfig,
        AtomVQVAEModule=AtomVQVAEModule,
        LigandAtomDescriptor=LigandAtomDescriptor,
        ProteinAtomDescriptor=ProteinAtomDescriptor,
        atom_descriptor_to_coords=atom_descriptor_to_coords,
        precompute_receptor_atom_features=precompute_receptor_atom_features,
        ATOM_DESCRIPTOR_DIM=ATOM_DESCRIPTOR_DIM,
        ATOM_LAYOUT=ATOM_LAYOUT,
        LIGAND_ELEMENT_VOCAB=LIGAND_ELEMENT_VOCAB,
        fields_by_name=fields_by_name,
        cartesian_to_spherical_np=cartesian_to_spherical_np,
        spherical_to_cartesian_np=spherical_to_cartesian_np,
        parse_sdf=parse_sdf,
        compute_canonical_frame=compute_canonical_frame,
        extract_pocket_atoms_from_candidates=extract_pocket_atoms_from_candidates,
        precompute_pocket_atom_candidates=precompute_pocket_atom_candidates,
    )


# --------------------------------------------------------------------------
# Pose quantization (ligand-own-frame arms)
# --------------------------------------------------------------------------


def _rotation_grid(n_rot: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n_rot, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([
            0.25 / s,
            (rot[2, 1] - rot[1, 2]) * s,
            (rot[0, 2] - rot[2, 0]) * s,
            (rot[1, 0] - rot[0, 1]) * s,
        ])
    else:
        i = int(np.argmax(np.diag(rot)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k])
        q = np.zeros(4)
        q[0] = (rot[k, j] - rot[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (rot[j, i] + rot[i, j]) / s
        q[k + 1] = (rot[k, i] + rot[i, k]) / s
    return q / np.linalg.norm(q)


def quantize_pose(centroid, rotation, box_origin, box_size, pose_bits, seed=0):
    """Quantize a rigid transform to ``pose_bits`` bits (half translation, half
    rotation). ``pose_bits=None`` returns it unchanged: oracle placement."""
    if pose_bits is None:
        return centroid, rotation
    trans_bits = pose_bits // 2
    rot_bits = pose_bits - trans_bits
    steps = max(int(round(2 ** (trans_bits / 3.0))), 1)
    cell = box_size / steps
    idx = np.clip(np.floor((centroid - box_origin) / cell), 0, steps - 1)
    centroid_q = box_origin + (idx + 0.5) * cell
    grid = _rotation_grid(2**rot_bits, seed)
    best = int(np.argmax(np.abs(grid @ _matrix_to_quat(rotation))))
    return centroid_q, _quat_to_matrix(grid[best])


# --------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------


class GridQuantizer:
    """Naive (element, spatial cell) tokenizer -- the no-learning reference.

    Presents the same ``encode`` / ``decode_to_outputs`` surface as a trained VQ.
    Answers "is the learned codebook actually buying anything over discretizing
    space at the same rate?".
    """

    def __init__(self, own, cells_per_axis: int = 10, box: float = 32.0) -> None:
        self.own = own
        self.cells = cells_per_axis
        self.box = box
        self.n_elements = len(own["LIGAND_ELEMENT_VOCAB"])
        fields = own["fields_by_name"](own["ATOM_LAYOUT"])
        self._coord = fields["coord"]
        self._element = fields["element"].start

    @property
    def codebook_size(self) -> int:
        return self.cells**3 * self.n_elements

    def _cell_index(self, cartesian: np.ndarray) -> np.ndarray:
        half = self.box / 2.0
        idx = np.floor((cartesian + half) / self.box * self.cells)
        return np.clip(idx, 0, self.cells - 1).astype(np.int64)

    def encode(self, x):
        import torch

        desc = x.cpu().numpy()
        sph = desc[:, self._coord.start : self._coord.end].astype(np.float64)
        grid = self._cell_index(self.own["spherical_to_cartesian_np"](sph))
        flat = (grid[:, 0] * self.cells + grid[:, 1]) * self.cells + grid[:, 2]
        element = desc[:, self._element].astype(np.int64)
        return torch.from_numpy(flat * self.n_elements + element).to(x.device)

    def decode_to_outputs(self, indices):
        import torch

        codes = indices.cpu().numpy()
        element = codes % self.n_elements
        flat = codes // self.n_elements
        grid = np.stack(
            [flat // self.cells**2, (flat // self.cells) % self.cells, flat % self.cells],
            axis=-1,
        ).astype(np.float64)
        half, step = self.box / 2.0, self.box / self.cells
        coord = self.own["cartesian_to_spherical_np"]((grid + 0.5) * step - half)
        logits = np.zeros((len(codes), self.n_elements), dtype=np.float32)
        logits[np.arange(len(codes)), element] = 1.0
        return {
            "coord": torch.from_numpy(coord).float().to(indices.device),
            "element": torch.from_numpy(logits).to(indices.device),
        }


def load_side(own, device, ckpt: str | None, norm: str | None, kind: str):
    """Build one modality's tokenizer + its normalization tensors."""
    import torch

    if kind == "binning":
        dim = own["ATOM_DESCRIPTOR_DIM"]
        grid = GridQuantizer(own)
        return {
            "vq": grid,
            "mean": torch.zeros(dim, device=device),
            "std": torch.ones(dim, device=device),
            "codebook_size": grid.codebook_size,
        }
    module = own["AtomVQVAEModule"].load_from_checkpoint(ckpt, map_location=device)
    module.eval().to(device)
    stats = torch.load(norm, weights_only=False)
    return {
        "vq": module.vqvae,
        "mean": stats["atom_mean"].to(device).float(),
        "std": stats["atom_std"].to(device).float(),
        "codebook_size": module.config.atom.codebook_size,
    }


def encode_decode(own, side, desc_np, meta, frame=None):
    """Descriptors -> tokens -> 3D coordinates. ``frame`` overrides the frame the
    decoded canonical coordinates are mapped back through."""
    import torch

    coord_f = own["fields_by_name"](own["ATOM_LAYOUT"])["coord"]
    mean_t, std_t = side["mean"], side["std"]
    desc_t = torch.from_numpy(desc_np).to(mean_t.device).float()
    with torch.no_grad():
        indices = side["vq"].encode((desc_t - mean_t) / std_t)
        outs = side["vq"].decode_to_outputs(indices)
    coord = outs["coord"] * std_t[coord_f.start : coord_f.end] + mean_t[coord_f.start : coord_f.end]
    recon = np.zeros((desc_np.shape[0], own["ATOM_DESCRIPTOR_DIM"]), dtype=np.float32)
    recon[:, coord_f.start : coord_f.end] = coord.cpu().numpy()
    coords = own["atom_descriptor_to_coords"](recon, meta, pocket_frame=frame)
    return coords, indices.cpu().numpy()


def load_receptor(own, receptor: Path, cache_dir: Path | None):
    """Parse a receptor once and reuse it forever.

    Parsing a full CASP protein costs ~10 s, which at 303 complexes is ~51 min --
    and every arm is a separate process, so without a cache on disk the same
    proteins get re-parsed once per arm (9 arms => 7.5 h of pure re-parsing, the
    single largest cost in the benchmark). Unpickling the parsed form takes 25 ms.
    """
    if cache_dir is None:
        return (
            own["precompute_pocket_atom_candidates"](receptor),
            own["precompute_receptor_atom_features"](receptor),
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.blake2b(str(receptor.resolve()).encode(), digest_size=16).hexdigest()
    path = cache_dir / f"{key}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 - our own cache, written below
    parsed = (
        own["precompute_pocket_atom_candidates"](receptor),
        own["precompute_receptor_atom_features"](receptor),
    )
    # Write via a temp file: a job killed mid-write must not leave a truncated
    # pickle that every later run then fails on.
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(parsed, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    return parsed


# The per-complex work splits cleanly in two: a CPU half (parse the SDF, extract
# the pocket, build descriptors) that dominates the wall clock, and a GPU half
# (encode/decode) that is milliseconds. Only the CPU half is parallelized -- the
# tokenizer stays in the parent process, so the GPU is neither duplicated across
# workers nor a source of contention.

_MAX_PREP_WORKERS = 16

_W: dict = {}


def _prep_init(workdir: str, receptor_cache: str | None) -> None:
    """Per-worker setup: import the source repo once, not once per complex."""
    _W["own"] = load_own(Path(workdir))
    _W["cache"] = Path(receptor_cache) if receptor_cache else None
    _W["prot_desc"] = _W["own"]["ProteinAtomDescriptor"]()
    _W["lig_desc"] = _W["own"]["LigandAtomDescriptor"]()
    _W["pocket_cfg"] = _W["own"]["PocketExtractionConfig"](max_residues=50)


def prepare_pair(task):
    """CPU half: everything needed to encode one complex, as plain arrays.

    Returns ``(tag, payload)`` with ``payload=None`` when the complex is
    unusable (no ligand heavy atoms, or no pocket within the cutoff).
    """
    tag, receptor, ligand, ligand_frame = task
    own = _W["own"]
    molecules = own["parse_sdf"](Path(ligand))
    if not molecules:
        return tag, None
    mol = molecules[0]
    heavy = np.array(
        [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"], dtype=np.float32
    )
    if len(heavy) == 0:
        return tag, None

    precomp, feats = load_receptor(own, Path(receptor), _W["cache"])
    pocket = own["extract_pocket_atoms_from_candidates"](precomp, heavy, _W["pocket_cfg"])
    if pocket is None or pocket.atom_coords.shape[0] == 0:
        return tag, None

    frame = own["compute_canonical_frame"](pocket.ca_coords.astype(np.float64))
    pdesc, pmeta = _W["prot_desc"].compute(pocket, feats, frame)
    lig_frame = (
        own["compute_canonical_frame"](heavy.astype(np.float64))
        if ligand_frame == "local"
        else frame
    )
    ldesc, _elements, lmeta = _W["lig_desc"].compute(
        mol["atoms"], mol["bonds"], lig_frame
    )
    if len(pdesc) == 0 or len(ldesc) == 0:
        return tag, None

    heavy_to_orig = list(lmeta["heavy_to_orig"])
    orig_to_heavy = {o: h for h, o in enumerate(heavy_to_orig)}
    return tag, {
        "pdesc": pdesc,
        "pmeta": pmeta,
        "ldesc": ldesc,
        "lmeta": lmeta,
        "protein_ref": pocket.atom_coords.astype(np.float64),
        "protein_elements": np.array(pocket.atom_elements, dtype="<U2"),
        "protein_atom_names": np.array(pocket.atom_names, dtype="<U4"),
        "protein_chain": np.array(pocket.atom_chain, dtype="<U4"),
        "protein_resid": np.array(pocket.atom_resseq, dtype=np.int64),
        "ligand_ref": np.array(
            [
                (mol["atoms"][i][1], mol["atoms"][i][2], mol["atoms"][i][3])
                for i in heavy_to_orig
            ],
            dtype=np.float64,
        ),
        "ligand_elements": np.array(
            [mol["atoms"][i][0] for i in heavy_to_orig], dtype="<U2"
        ),
        "ligand_bonds": np.array(
            [
                (orig_to_heavy[b[0]], orig_to_heavy[b[1]], b[2])
                for b in mol["bonds"]
                if b[0] in orig_to_heavy and b[1] in orig_to_heavy
            ],
            dtype=np.int64,
        ).reshape(-1, 3),
    }


def encode_prepared(own, arm, sides, prep):
    """GPU half: quantize the prepared descriptors and assemble the NPZ payload."""
    prot_rec, prot_idx = encode_decode(own, sides["protein"], prep["pdesc"], prep["pmeta"])
    if arm["ligand_frame"] == "local":
        box_origin = prep["protein_ref"].min(axis=0)
        box_size = float((prep["protein_ref"].max(axis=0) - box_origin).max())
        placed = quantize_pose(
            prep["lmeta"]["centroid"], prep["lmeta"]["rotation"],
            box_origin, box_size, arm["pose_bits"],
        )
        lig_rec, lig_idx = encode_decode(
            own, sides["ligand"], prep["ldesc"], prep["lmeta"], frame=placed
        )
    else:
        lig_rec, lig_idx = encode_decode(own, sides["ligand"], prep["ldesc"], prep["lmeta"])

    keep = (
        "protein_ref", "protein_elements", "protein_atom_names",
        "protein_chain", "protein_resid", "ligand_ref", "ligand_elements",
        "ligand_bonds",
    )
    return {
        **{k: prep[k] for k in keep},
        "protein_rec": np.asarray(prot_rec, dtype=np.float64),
        "ligand_rec": np.asarray(lig_rec, dtype=np.float64),
        "n_tokens_protein": np.int64(len(prot_idx)),
        "n_tokens_ligand": np.int64(len(lig_idx)),
        "bits_protein": np.float64(np.log2(sides["protein"]["codebook_size"])),
        "bits_ligand": np.float64(np.log2(sides["ligand"]["codebook_size"])),
        "pose_bits": np.float64(arm["pose_bits"] or 0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--arm", type=Path, required=True, help="arm spec JSON")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="CPU workers for the descriptor stage (default: allocated cores - 1)",
    )
    p.add_argument(
        "--receptor-cache",
        type=Path,
        default=None,
        help="directory for pickled parsed receptors, shared across arms and "
        "runs. Without it every arm re-parses every protein (~10 s each).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="torch device; defaults to cuda when available. Pass 'cpu' to run "
        "while the GPU is occupied -- reconstruction is small enough that this "
        "is slow but workable.",
    )
    args = p.parse_args()

    import torch

    own = load_own(args.workdir)
    arm = json.loads(args.arm.read_text())
    arm.setdefault("ligand_frame", "pocket")
    arm.setdefault("pose_bits", None)
    arm.setdefault("kind", "vq")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    sides = {
        side: load_side(
            own, device, arm.get(f"{side}_ckpt"), arm.get(f"{side}_norm"), arm["kind"]
        )
        for side in ("protein", "ligand")
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(args.pairs.read_text())
    tasks = [
        (p["id"], p["receptor"], p["ligand"], arm["ligand_frame"]) for p in pairs
    ]

    # Pool size from the affinity mask, not cpu_count: on a scheduler-allocated
    # node cpu_count reports the whole machine while the job owns a slice, and
    # oversubscribing by that factor is far slower than running serially.
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 2)
    )
    # Capped: every worker imports torch and the source repo (~0.8 GB RSS each),
    # so scaling to all 48 cores of a node_q would spend more on memory than the
    # remaining speedup is worth -- an unbounded pool is what got a local test
    # OOM-killed. The descriptor stage is ~4 s/complex, so 16 workers already
    # brings 303 complexes down to about a minute.
    default_workers = min(available - 1, _MAX_PREP_WORKERS)
    workers = max(1, min(args.workers or default_workers, len(tasks)))
    print(f"[own-allatom] preparing {len(tasks)} complexes on {workers} workers")

    summary, n_ok = [], 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        workers, initializer=_prep_init,
        initargs=(str(args.workdir), str(args.receptor_cache) if args.receptor_cache else None),
    ) as pool:
        for tag, prep in pool.imap_unordered(prepare_pair, tasks, chunksize=1):
            if prep is None:
                summary.append({"id": tag, "ok": False, "error": "no pocket / empty ligand"})
                continue
            try:
                # Quantization runs in the parent: the tokenizer lives on one GPU
                # and is far too fast to be worth distributing.
                out = encode_prepared(own, arm, sides, prep)
            except Exception as exc:  # noqa: BLE001 - one bad complex must not stop the run
                summary.append({"id": tag, "ok": False, "error": repr(exc)})
                print(f"[own-allatom] skip {tag}: {exc}", file=sys.stderr)
                continue
            np.savez_compressed(args.out_dir / f"{tag}.npz", **out)
            summary.append({"id": tag, "ok": True})
            n_ok += 1

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[own-allatom] reconstructed {n_ok}/{len(pairs)} complexes -> {args.out_dir}")


if __name__ == "__main__":
    main()
