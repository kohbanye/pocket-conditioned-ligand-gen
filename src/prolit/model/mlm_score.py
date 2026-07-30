"""Pseudo-log-likelihood pose scoring with the complex-token MLM.

The bidirectional MLM (:class:`~prolit.model.complex_mlm.ComplexMLM`) is turned into
a pose rescorer via masked pseudo-log-likelihood (PLL): each ligand token is
masked in turn and the model's log-probability of the true token (given the
pocket + the rest of the ligand, bidirectionally) is read off. Summed over the
ligand span this is a Besag pseudo-likelihood of ``P(ligand pose | pocket)`` --
a native-like pose sits in a high-probability region of the learnt binding-mode
distribution, a decoy does not.

Length-normalised (mean over ligand positions) by default to avoid the
ligand-size bias that plagues raw distance-likelihood scores (cf. NMDN).
"""

from __future__ import annotations

import torch

from prolit.tokenizers.lm_vocab import L_CLOSE_ID, L_OPEN_ID, NUM_SPECIAL


def ligand_positions(input_ids: list[int]) -> list[int]:
    """Indices of the ligand codebook tokens (strictly inside ``<l>..</l>``)."""
    try:
        lo = input_ids.index(L_OPEN_ID)
        hi = len(input_ids) - 1 - input_ids[::-1].index(L_CLOSE_ID)
    except ValueError:
        return []
    return [p for p in range(lo + 1, hi) if input_ids[p] >= NUM_SPECIAL]


@torch.no_grad()
def ligand_pll(  # noqa: PLR0913
    model: torch.nn.Module,
    input_ids: list[int],
    mask_token_id: int,
    device: torch.device,
    *,
    normalize: bool = True,
    mask_batch: int = 64,
) -> float:
    """Masked PLL of the ligand span. Higher = more native-like.

    One encoder pass per ``mask_batch`` ligand positions (each row masks a
    distinct position), so a ~30-atom ligand costs one forward. Returns the mean
    (``normalize``) or sum of ``log P(true token | masked)`` over ligand tokens.
    """
    positions = ligand_positions(input_ids)
    if not positions:
        return float("nan")
    base = torch.tensor(input_ids, dtype=torch.long, device=device)
    total = 0.0
    for start in range(0, len(positions), mask_batch):
        chunk = positions[start : start + mask_batch]
        batch = base.unsqueeze(0).repeat(len(chunk), 1).clone()
        rows = torch.arange(len(chunk))
        cols = torch.tensor(chunk)
        batch[rows, cols] = mask_token_id
        out = model(input_ids=batch, attention_mask=torch.ones_like(batch))
        logp = torch.log_softmax(out.logits.float(), dim=-1)
        total += logp[rows, cols, base[cols]].sum().item()
    return total / len(positions) if normalize else total
