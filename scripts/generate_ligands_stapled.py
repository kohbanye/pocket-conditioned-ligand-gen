"""Pocket-conditioned generation for the ESM3 x ConfSeq baseline -> generated.sdf.

The stapled counterpart of ``generate_ligands_for_target.py``, kept separate
rather than folded in as another mode. The two share the prompt-and-sample half
and nothing else: ProLIT decodes atom tokens through a VQ-VAE into a shared
pocket frame, while this decodes ConfSeq tokens into the molecule's OWN frame
and then places that rigid body with the four quantized pose tokens. Threading
a second decoder through the generator that produces the paper's ProLIT numbers
would put those numbers at risk for no gain.

Emits the layout the sbdd-bench ``own`` adapter reads::

    generated.sdf   every decoded ligand as a heavy-atom mol block

The adapter decides success on this process's EXIT CODE, not on the file, so
every unrecoverable condition here exits non-zero rather than writing an empty
result that would be scored as "the baseline generated nothing valid".

Run (driven by the adapter; this form is for one target by hand)::

    PYTHONPATH=$PWD .venv/bin/python scripts/generate_ligands_stapled.py \
        --receptor <target>_receptor.pdb --ref-ligand <target>_ref_ligand.sdf \
        --target-id ABL2_HUMAN_274_551_0 --lm-ckpt <clm.ckpt> \
        --esm3-cache data/esm3_tokens_sbdd_targets \
        --stapled-vocab data/stapled/confseq_vocab.json \
        --num-samples 100 --out-dir outputs/abl2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from generate_ligands_3d import load_atom_lm  # noqa: E402

from prolit.config import PocketExtractionConfig  # noqa: E402
from prolit.data.esm3_tokens import Esm3TokenCache  # noqa: E402
from prolit.seeding import add_seed_argument, seed_from_args  # noqa: E402
from prolit.tokenizers.ligand import parse_sdf  # noqa: E402
from prolit.tokenizers.lm_vocab import (  # noqa: E402
    BOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    P_CLOSE_ID,
    P_OPEN_ID,
    PAD_ID,
)
from prolit.tokenizers.stapled import (  # noqa: E402
    ConfSeqVocab,
    StapledVocab,
    confseq_decode,
    place,
)
from prolit.tokenizers.stapled_encoder import StapledEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _write_sdf(path: Path, mols: list) -> None:
    """One heavy-atom mol block per generated ligand."""
    from rdkit import Chem  # noqa: PLC0415

    with Chem.SDWriter(str(path)) as w:
        for i, m in enumerate(mols):
            m.SetProp("_Name", f"gen_{i}")
            w.write(m)


def main() -> None:  # noqa: PLR0915
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receptor", type=Path, required=True)
    p.add_argument("--ref-ligand", type=Path, required=True)
    p.add_argument(
        "--target-id",
        default=None,
        help="Key into the ESM3 cache. Defaults to the receptor's parent "
        "directory name, which is how sbdd-bench lays its targets out.",
    )
    p.add_argument("--lm-ckpt", type=Path, required=True)
    p.add_argument("--esm3-cache", type=Path, required=True)
    p.add_argument("--stapled-vocab", type=Path, required=True)
    p.add_argument("--confseq-repo", type=Path, default=_REPO / "third_party/ConfSeq")
    p.add_argument("--atom-codebook-size", type=int, default=12733)
    p.add_argument("--max-residues", type=int, default=50)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out-dir", type=Path, required=True)
    add_seed_argument(p)
    args = p.parse_args()
    seed_from_args(args)

    target_id = args.target_id or args.receptor.resolve().parent.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = StapledVocab(confseq=ConfSeqVocab.load(args.stapled_vocab))
    enc = StapledEncoder(
        cache=Esm3TokenCache(args.esm3_cache),
        confseq_repo=args.confseq_repo,
        vocab=vocab,
        pocket_cfg=PocketExtractionConfig(max_residues=args.max_residues),
    )

    ref = parse_sdf(args.ref_ligand)[0]
    ref_heavy = np.array(
        [(a[1], a[2], a[3]) for a in ref["atoms"] if a[0] != "H"], dtype=np.float32
    )
    pocket = enc.setup_pocket(target_id, args.receptor.read_text(), ref_heavy)
    if pocket is None:
        # Non-zero, loudly: a silent empty SDF here is indistinguishable from
        # "the model generated nothing valid", and would be scored as such.
        logger.error("no pocket for %s (ESM3 cache miss or empty pocket)", target_id)
        raise SystemExit(2)

    model = load_atom_lm(str(args.lm_ckpt), args.atom_codebook_size, device)
    prompt = [BOS_ID, P_OPEN_ID, *(vocab.esm3_offset + int(c) for c in pocket.codes)]
    prompt += [P_CLOSE_ID, L_OPEN_ID]
    logger.info(
        "%s: %d pocket codes, prompt %d tokens",
        target_id,
        len(pocket.codes),
        len(prompt),
    )

    mols: list = []
    n_pose_missing = n_decode_fail = n_attempt = 0
    while len(mols) < args.num_samples and n_attempt < args.num_samples * 4:
        batch = min(args.batch_size, args.num_samples - len(mols))
        n_attempt += batch
        ids = torch.tensor([prompt] * batch, dtype=torch.long, device=device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=L_CLOSE_ID,
                pad_token_id=PAD_ID,
            )
        for k in range(gen.shape[0]):
            seq = gen[k].tolist()
            _esm3, pose, confseq_ids = vocab.split_sequence(seq)
            if pose is None:
                # The four placement tokens lead the ligand block, so a
                # truncated sample loses the placement rather than getting a
                # wrong one. Counted, not silently centred.
                n_pose_missing += 1
                continue
            if not confseq_ids:
                n_decode_fail += 1
                continue
            mol = confseq_decode(vocab.confseq.strings(confseq_ids), args.confseq_repo)
            if mol is None:
                n_decode_fail += 1
                continue
            own = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
            placed = place(own, pose, pocket.backbone)
            conf = mol.GetConformer()
            for i, (x, y, z) in enumerate(placed):
                conf.SetAtomPosition(i, (float(x), float(y), float(z)))
            mols.append(mol)
            if len(mols) >= args.num_samples:
                break

    _write_sdf(args.out_dir / "generated.sdf", mols)
    logger.info(
        "%s: %d/%d generated (%d attempts; %d lost the pose, %d failed decode)",
        target_id,
        len(mols),
        args.num_samples,
        n_attempt,
        n_pose_missing,
        n_decode_fail,
    )
    if not mols:
        logger.error("%s: nothing decoded", target_id)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
