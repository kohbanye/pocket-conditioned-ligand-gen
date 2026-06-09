"""Conditional ligand-token generation test for the pocket-conditioned LM.

Loads a trained LM checkpoint, takes real protein pockets from the held-out
*test* split, prompts the model with ``<bos><p> pocket... </p><l>`` and
autoregressively samples ligand structure tokens until ``</l>``. Reports
validity (clean termination, tokens in the ligand range), generated vs
ground-truth length, and sample diversity.

This checks the LM half of the pipeline (does it emit coherent, pocket-
conditioned ligand token streams). Decoding tokens back to 3D coordinates via
the VQ-VAE ligand decoder is a separate step.

Run on a GPU node::

    uv run python scripts/generate_ligands.py \
        --ckpt "pocket-ligand-lm/ao4pqv5u/checkpoints/lm-epoch=01-val/loss=1.5521.ckpt" \
        --num-pockets 5 --num-samples 3
"""
# Eval CLI: stdout report tables (T201), inline preview counts (PLR2004) and
# dense report f-strings (E501) are intentional here.
# ruff: noqa: T201, PLR2004, E501

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src.config import LMTrainingConfig
from src.model.lm_module import LigandLMModule
from src.tokenizers.lm_vocab import (
    L_CLOSE_ID,
    L_OPEN_ID,
    PAD_ID,
    LMVocab,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_test_docs(token_dir: Path, num_docs: int, seed: int) -> list[list[int]]:
    """Return token-id lists for ``num_docs`` random test documents."""
    lens = np.fromfile(token_dir / "test.len", dtype=np.uint16).astype(np.int64)
    bin_ = np.memmap(token_dir / "test.bin", dtype=np.uint16, mode="r")
    offsets = np.concatenate([[0], np.cumsum(lens)])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lens), size=min(num_docs, len(lens)), replace=False)
    return [bin_[offsets[i] : offsets[i + 1]].tolist() for i in idx]


def _pocket_prompt(doc: list[int]) -> tuple[list[int], list[int]]:
    """Split a full doc into (prompt up to and incl. ``<l>``, gt ligand codes)."""
    l_open = doc.index(L_OPEN_ID)
    prompt = doc[: l_open + 1]
    # Ground-truth ligand tokens are between <l> and </l>.
    gt_tail = doc[l_open + 1 :]
    gt_ligand = (
        gt_tail[: gt_tail.index(L_CLOSE_ID)] if L_CLOSE_ID in gt_tail else gt_tail
    )
    return prompt, gt_ligand


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--token-dir", type=Path, default=Path("data/lm_tokens"))
    parser.add_argument("--num-pockets", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    vocab = LMVocab()
    lig_lo, lig_hi = (
        vocab.ligand_offset,
        vocab.ligand_offset + vocab.ligand_codebook_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = LigandLMModule.load_from_checkpoint(
        args.ckpt, config=LMTrainingConfig(), map_location=device
    )
    module.eval()
    module.to(device)
    model = module.model

    docs = _load_test_docs(args.token_dir, args.num_pockets, args.seed)
    logger.info("Loaded %d test pockets from %s", len(docs), args.token_dir)

    valid_term = 0
    total = 0
    gen_lens: list[int] = []
    gt_lens: list[int] = []

    for pi, doc in enumerate(docs):
        prompt, gt_ligand = _pocket_prompt(doc)
        n_pocket = sum(
            1 for t in prompt if vocab.protein_offset <= t < vocab.ligand_offset
        )
        prompt_ids = torch.tensor([prompt], device=device)
        attn = torch.ones_like(prompt_ids)
        with torch.no_grad():
            out = model.generate(
                input_ids=prompt_ids.repeat(args.num_samples, 1),
                attention_mask=attn.repeat(args.num_samples, 1),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=L_CLOSE_ID,
                pad_token_id=PAD_ID,
            )
        print(
            f"\n=== pocket {pi} | {n_pocket} residues | GT ligand {len(gt_ligand)} atoms ==="
        )
        print(
            "  GT  : "
            + " ".join(f"L{t - lig_lo}" for t in gt_ligand[:24])
            + (" ..." if len(gt_ligand) > 24 else "")
        )
        gt_lens.append(len(gt_ligand))
        seen = set()
        for si in range(out.shape[0]):
            gen = out[si].tolist()[len(prompt) :]
            terminated = L_CLOSE_ID in gen
            lig = gen[: gen.index(L_CLOSE_ID)] if terminated else gen
            in_range = all(lig_lo <= t < lig_hi for t in lig)
            ok = terminated and in_range and len(lig) > 0
            valid_term += int(ok)
            total += 1
            gen_lens.append(len(lig))
            seen.add(tuple(lig))
            flag = "ok " if ok else ("noend" if not terminated else "oor")
            codes = " ".join(
                f"L{t - lig_lo}" if lig_lo <= t < lig_hi else f"?{t}" for t in lig[:24]
            )
            print(
                f"  s{si}[{flag} len={len(lig):3d}]: {codes}"
                + (" ..." if len(lig) > 24 else "")
            )
        print(f"  distinct samples: {len(seen)}/{out.shape[0]}")

    print("\n================ SUMMARY ================")
    print(
        f"valid (terminated + in-range + nonempty): {valid_term}/{total} = {100 * valid_term / max(total, 1):.0f}%"
    )
    print(
        f"mean generated ligand len: {np.mean(gen_lens):.1f}  (GT mean: {np.mean(gt_lens):.1f})"
    )
    print(f"gen len range: {min(gen_lens)}-{max(gen_lens)}")


if __name__ == "__main__":
    main()
