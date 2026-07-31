"""Dense Qwen3 decoder builder for the pocket-conditioned ligand LM.

Builds a randomly-initialized ``Qwen3ForCausalLM`` (full attention, GQA,
RMSNorm, SwiGLU, RoPE, QK-Norm) sized from :class:`~prolit.config.ProLITCLMConfig`.
This is a from-scratch model — no pretrained weights are loaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import Qwen3Config, Qwen3ForCausalLM

from prolit.tokenizers.lm_vocab import BOS_ID, EOS_ID, PAD_ID

if TYPE_CHECKING:
    from prolit.config import ProLITCLMConfig


def build_qwen3_config(cfg: ProLITCLMConfig) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_parameters={  # ty: ignore[missing-typed-dict-key]
            "rope_type": "default",
            "rope_theta": cfg.rope_theta,
        },
        rms_norm_eps=cfg.rms_norm_eps,
        hidden_act=cfg.hidden_act,
        attention_bias=cfg.attention_bias,
        attention_dropout=cfg.attention_dropout,
        tie_word_embeddings=cfg.tie_word_embeddings,
        pad_token_id=PAD_ID,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        use_cache=False,
    )


def build_qwen3_lm(cfg: ProLITCLMConfig) -> Qwen3ForCausalLM:
    """Construct a randomly-initialized Qwen3 causal LM from scratch."""
    hf_config = build_qwen3_config(cfg)
    return Qwen3ForCausalLM._from_config(  # noqa: SLF001
        hf_config,
        attn_implementation=cfg.attn_implementation,
    )


def count_parameters(model: Qwen3ForCausalLM) -> int:
    return sum(p.numel() for p in model.parameters())
