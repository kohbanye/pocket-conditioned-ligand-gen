"""Self-contained smoke test for the ligand LM training stack.

Generates a tiny synthetic token cache (random protein/ligand codebook
sequences assembled via :mod:`src.tokenizers.lm_vocab`), then overfits a tiny
Qwen3 model on it for a few hundred steps. Validates end to end:

- the unified vocabulary + packed dataset + block-diagonal 4D attention mask,
- that ``Qwen3ForCausalLM`` accepts our custom mask under the chosen backend,
- that loss falls from ~log(vocab) toward ~0 (i.e. the model can fit).

Run::

    uv run python scripts/smoke_train_lm.py
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from pathlib import Path

import lightning as L
import numpy as np
import torch

from src.config import LMTrainingConfig
from src.data.lm_dataset import LMTokenDataModule
from src.model.lm_module import LigandLMModule
from src.tokenizers.lm_vocab import LMVocab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RNG = np.random.default_rng(0)


def _write_synthetic_cache(  # noqa: PLR0913
    out_dir: Path,
    vocab: LMVocab,
    num_docs: int,
    n_pocket: int,
    n_ligand: int,
    splits: tuple[str, ...] = ("train", "val"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "vocab_size": vocab.vocab_size,
        "protein_codebook_size": vocab.protein_codebook_size,
        "ligand_codebook_size": vocab.ligand_codebook_size,
        "protein_offset": vocab.protein_offset,
        "ligand_offset": vocab.ligand_offset,
        "splits": {},
    }
    for split in splits:
        all_tokens: list[int] = []
        lengths: list[int] = []
        for _ in range(num_docs):
            p = RNG.integers(0, vocab.protein_codebook_size, size=n_pocket).tolist()
            lig = RNG.integers(0, vocab.ligand_codebook_size, size=n_ligand).tolist()
            seq = vocab.build_sequence(p, lig)
            all_tokens.extend(seq)
            lengths.append(len(seq))
        np.asarray(all_tokens, dtype=np.uint16).tofile(out_dir / f"{split}.bin")
        np.asarray(lengths, dtype=np.uint16).tofile(out_dir / f"{split}.len")
        meta["splits"][split] = {
            "num_docs": num_docs,
            "num_tokens": len(all_tokens),
            "max_len": max(lengths),
        }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    vocab = LMVocab(protein_codebook_size=256, ligand_codebook_size=128)

    with tempfile.TemporaryDirectory() as tmp:
        token_dir = Path(tmp) / "tokens"
        # 32 distinct docs, repeated by epochs -> the tiny model should overfit.
        _write_synthetic_cache(
            token_dir, vocab, num_docs=32, n_pocket=12, n_ligand=10
        )

        config = LMTrainingConfig()
        config.token_dir = token_dir
        config.block_size = 128
        config.micro_batch_size = 8
        config.num_workers = 0
        config.warmup_steps = 20
        config.learning_rate = 3e-3
        config.max_epochs = 60
        config.precision = "32-true"
        # Tiny model.
        config.model.protein_codebook_size = vocab.protein_codebook_size
        config.model.ligand_codebook_size = vocab.ligand_codebook_size
        config.model.hidden_size = 128
        config.model.num_hidden_layers = 2
        config.model.num_attention_heads = 4
        config.model.num_key_value_heads = 2
        config.model.head_dim = 32
        config.model.intermediate_size = 256
        config.model.max_position_embeddings = 128

        dm = LMTokenDataModule(config)
        module = LigandLMModule(config)

        trainer = L.Trainer(
            max_epochs=config.max_epochs,
            accelerator="auto",
            devices=1,
            precision=config.precision,
            gradient_clip_val=config.grad_clip,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=1,
        )
        trainer.fit(module, dm)

        # Evaluate final train loss to confirm the model fit the data.
        module.eval()
        dm.setup()
        losses = []
        with torch.no_grad():
            for batch in dm.train_dataloader():
                dev_batch = {k: v.to(module.device) for k, v in batch.items()}
                losses.append(module(dev_batch).item())
        final_loss = float(np.mean(losses))
        baseline = math.log(vocab.vocab_size)
        logger.info(
            "Smoke result: final_train_loss=%.4f (baseline log(vocab)=%.4f)",
            final_loss,
            baseline,
        )
        if final_loss > 1.0:
            msg = (
                f"Overfit smoke FAILED: final loss {final_loss:.3f} did not drop "
                f"below 1.0 (baseline {baseline:.3f}). Check the model/mask plumbing."
            )
            raise SystemExit(msg)
        logger.info("Smoke PASSED: model overfit the synthetic data as expected.")


if __name__ == "__main__":
    main()
