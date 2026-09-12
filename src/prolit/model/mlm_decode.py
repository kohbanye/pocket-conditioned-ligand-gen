"""Let the complex MLM revise the language model's ligand codes.

The causal model has to commit to each atom before it has seen the rest of the
molecule, and its uncertainty says exactly what that costs. Measured as the
spatial spread of its predictive distribution over the code's decoded position:

    conditioning                                       spread
    pocket only (what the first atom gets)              5.30 A
    pocket + the atoms already emitted (autoregressive) 1.40 A
    pocket + every other atom (bidirectional)           0.65 A

So the anchor is chosen while the model is still 5.30 A uncertain, and every
later atom inherits it -- the first atom lands 3.19 A from the crystal against
~1.5 A for the rest. The true code is in the causal model's top 10 for 90.4% of
atoms but ranked first for only 47.8%: the candidate is there, the ranking is
not, and the information that would fix the ranking sits in the atoms it has
not written yet.

Re-ranking with the true suffix takes RMSD 1.070 -> 0.601 on 97.5% of
molecules; re-ranking with the causal model's *own* rollout makes it worse
(+0.130), which is why beam search is not the answer and a bidirectional model
is. Feeding the causal codes to a ligand-masked MLM and letting it re-decide the
positions it is least sure of takes 1.070 -> 0.857 on 67.5% of molecules.

Starting from a fully masked ligand instead ("cold") is much worse (4.97) even
with the mask rate randomised: generating a whole molecule bidirectionally is a
harder problem than repairing one, and one epoch does not reach it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from prolit.tokenizers.lm_vocab import NUM_SPECIAL

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    #: Given a ``(B, n)`` array of ligand code sequences, which positions of each
    #: are unacceptable. The optional second argument restricts the answer to one
    #: column, which is all the picker ever reads. Geometry stays in the caller:
    #: this layer never opens a receptor, holds a frame, or knows a radius.
    ClashProbe = Callable[..., np.ndarray]

#: Below this the ligand is a fragment and re-masking a fraction of it rounds to
#: the whole molecule, which is the cold case the model is worst at.
MIN_LIGAND_CODES = 4

#: Fewer unchanged atoms than this and a superposition is not determined.
MIN_ANCHOR_ATOMS = 3


@torch.no_grad()
def _logits(model: nn.Module, ids: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(ids[None]).to(device=device, dtype=torch.long)
    out = model(input_ids=t, attention_mask=torch.ones_like(t))
    return out.logits[0] if hasattr(out, "logits") else out[0]


@torch.no_grad()
def refine_codes(  # noqa: PLR0913
    model: nn.Module,
    mask_token_id: int,
    protein_codes: Sequence[int],
    ligand_codes: Sequence[int],
    *,
    codebook_size: int,
    rounds: int = 8,
    frac: float = 0.25,
    order: str = "confidence",
    clash_probe: ClashProbe | None = None,
    accept_probe: ClashProbe | None = None,
    candidates: int = 64,
    device: torch.device | None = None,
) -> list[int]:
    """Re-decide ligand codes, ``rounds`` times.

    ``order="confidence"`` masks the ``frac`` of positions the model assigns the
    lowest probability to *as they currently stand*. That is MaskGIT's schedule,
    and it is the wrong one here: the error this is meant to repair is not
    spread uniformly, it *accumulates along the decode order*. Measured over 40
    targets, the fraction of ligand atoms within 3.0 A of a protein atom climbs
    monotonically 11.4% -> 33.7% from the first tenth of the sequence to the
    last, while FLOWR -- which emits every atom at once -- is flat at 7.8% ->
    8.1%. Confidence masking left that slope untouched (12.2% -> 33.9%), which
    is why it moved Vina by 0.11 kcal and cost PoseBusters validity.

    ``order="late_first"`` instead sweeps contiguous blocks backwards from the
    end of the sequence, re-predicting each block with the earlier atoms -- the
    ones that have accumulated the least drift -- visible as an anchor. Blocks
    are ``frac`` of the sequence and ``rounds`` of them are swept, so
    ``rounds * frac`` is the tail fraction that gets re-derived.

    ``order="clash"`` re-decides exactly the positions whose atoms are inside the
    protein, and requires ``clash_probe``. It is the aimed version of
    ``late_first``: the clash RATE climbs 4.6% -> 12.8% along the token order on
    the deployed arm while the reference ligand is flat at 0.4% -> 1.0%
    (2026-09-10), so the late block is where clashes are dense -- but 87% of the
    atoms in it are fine, and re-deciding those is wasted risk.

    This is the only order that can be aimed correctly. A mask applied while the
    causal model is still emitting cannot be: the position the VQ decoder gives
    the newest atom from a left prefix is a median **0.930 A** from where that
    atom ends up once the sequence is complete, which is the same size as the
    clash criterion's own margin. With the whole ligand block on the table the
    decode is exact, so the probe sees where the atoms actually are.

    ``accept_probe`` is what a REPLACEMENT is judged by, and defaults to
    ``clash_probe``. They are different questions: a position is re-decided
    because it is in the wall, but a candidate is only acceptable if it is out
    of the wall *and* has not broken the molecule. Judging replacements by the
    wall alone was measured to cost exactly the PoseBusters it gained in score
    (13.4% of molecules outside the bond window against a control's 6.5%, and
    validity down 7.0 points).

    Replacement is drawn from the ``candidates`` most probable codes rather than
    the whole codebook, and the most probable acceptable one wins; if every
    candidate is rejected the position keeps the model's argmax, so the schedule
    can never fail to terminate. Scanning all 8192 would cost 128x
    more decode for a tail the model puts almost no mass on.

    Returns codebook indices, not vocabulary ids.
    """
    if order == "clash" and clash_probe is None:
        msg = "order='clash' needs a clash_probe; geometry does not live in this layer"
        raise ValueError(msg)
    codes = [int(c) for c in ligand_codes]
    n = len(codes)
    if n < MIN_LIGAND_CODES or rounds < 1 or frac <= 0:
        return codes
    if device is None:
        # next() on an empty generator raises StopIteration, which unwinds as a
        # confusing error rather than a missing-device one.
        first = next(iter(model.parameters()), None)
        device = first.device if first is not None else torch.device("cpu")

    from prolit.tokenizers.lm_vocab import (  # noqa: PLC0415
        BOS_ID,
        L_CLOSE_ID,
        L_OPEN_ID,
        P_CLOSE_ID,
        P_OPEN_ID,
    )

    head = [BOS_ID, P_OPEN_ID, *(NUM_SPECIAL + int(c) for c in protein_codes)]
    head += [P_CLOSE_ID, L_OPEN_ID]
    lo = len(head)
    body = [NUM_SPECIAL + c for c in codes]
    ids = np.array([*head, *body, L_CLOSE_ID], dtype=np.int64)
    hi = lo + n
    k = max(1, round(frac * n))

    for r in range(rounds):
        span = slice(NUM_SPECIAL, NUM_SPECIAL + codebook_size)
        weak = _select(
            order, codes, ids, model, device, span, lo, hi, k, r, n, clash_probe
        )
        if weak is None:
            break
        if weak.size == 0:
            continue
        ids[lo + weak] = mask_token_id
        probs = torch.softmax(_logits(model, ids, device)[lo:hi, span].float(), -1)
        if order == "clash":
            assert clash_probe is not None  # noqa: S101 -- checked above
            picked = _pick_clear(
                codes, weak, probs, accept_probe or clash_probe, candidates
            )
        else:
            picked = probs[weak].argmax(-1).cpu().numpy()
        for slot, code in zip(weak, picked, strict=True):
            codes[int(slot)] = int(code)
            ids[lo + int(slot)] = NUM_SPECIAL + int(code)
    return codes


def _select(  # noqa: PLR0913
    order: str,
    codes: list[int],
    ids: np.ndarray,
    model: nn.Module,
    device: torch.device,
    span: slice,
    lo: int,
    hi: int,
    k: int,
    r: int,
    n: int,
    clash_probe: ClashProbe | None,
) -> np.ndarray | None:
    """Which positions this round re-decides. ``None`` ends the schedule early."""
    if order == "clash":
        assert clash_probe is not None  # noqa: S101 -- checked by the caller
        bad = np.flatnonzero(clash_probe(np.asarray([codes]))[0])
        if bad.size == 0:
            return None  # nothing is in the wall; further rounds are only risk
        return bad[:k]
    if order == "late_first":
        # March a block of width k backwards from the tail, wrapping so a long
        # sweep re-derives the whole sequence rather than running off the front.
        end = n - (r * k) % n
        return np.arange(max(0, end - k), end)
    probs = torch.softmax(_logits(model, ids, device)[lo:hi, span].float(), -1)
    held = probs.gather(1, torch.tensor(codes, device=device)[:, None])
    return torch.argsort(held.squeeze(1))[:k].cpu().numpy()


def _pick_clear(
    codes: list[int],
    weak: np.ndarray,
    probs: torch.Tensor,
    accept: ClashProbe,
    candidates: int,
) -> np.ndarray:
    """The most probable candidate code the accept probe does not reject.

    One probe call per masked position, batched over that position's candidates.
    Falls back to the plain argmax when every candidate is rejected, so a buried
    position degrades to ordinary MaskGIT rather than blocking.
    """
    out = []
    for slot in weak:
        row = probs[int(slot)]
        m = min(candidates, row.shape[0])
        top = torch.topk(row, m).indices.cpu().numpy()
        trial = np.tile(np.asarray(codes, dtype=np.int64), (m, 1))
        trial[:, int(slot)] = top
        clear = ~accept(trial, int(slot))[:, int(slot)]
        out.append(int(top[int(np.argmax(clear))]) if clear.any() else int(top[0]))
    return np.asarray(out, dtype=np.int64)


def cold_decode(  # noqa: PLR0913
    model: nn.Module,
    mask_token_id: int,
    protein_codes: Sequence[int],
    n_ligand: int,
    *,
    codebook_size: int,
    rounds: int = 8,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> list[int]:
    """Decode a ligand of ``n_ligand`` codes from nothing, most-confident-first.

    :func:`refine_codes` is a *warm* start: it revises what the causal model
    already wrote, so every code it keeps still carries the left-to-right
    order's accumulated drift. This is the cold start -- every ligand position
    begins masked and the bidirectional model fills them in confidence order,
    so no position is ever committed on a left prefix alone.

    That ordering is the point. Measured against the crystal ligand under the
    same atom order, the excess clash rate grows +3.3 -> +18.4 points along the
    token sequence for causal generation, while the tokenizer's own round trip
    only grows -0.4 -> +6.0. Two thirds of the slope is the decode order, and
    this is the arm that removes it rather than patching it.

    ``rounds`` follows MaskGIT's cosine schedule: round ``t`` leaves
    ``n * cos(pi/2 * (t+1)/rounds)`` positions masked, so early rounds commit
    few, well-supported positions and later rounds fill the rest in their
    context.
    """
    n = int(n_ligand)
    if n < MIN_LIGAND_CODES or rounds < 1:
        return []
    if device is None:
        first = next(iter(model.parameters()), None)
        device = first.device if first is not None else torch.device("cpu")

    from prolit.tokenizers.lm_vocab import (  # noqa: PLC0415
        BOS_ID,
        L_CLOSE_ID,
        L_OPEN_ID,
        P_CLOSE_ID,
        P_OPEN_ID,
    )

    head = [BOS_ID, P_OPEN_ID, *(NUM_SPECIAL + int(c) for c in protein_codes)]
    head += [P_CLOSE_ID, L_OPEN_ID]
    lo = len(head)
    ids = np.array([*head, *([mask_token_id] * n), L_CLOSE_ID], dtype=np.int64)
    hi = lo + n
    span = slice(NUM_SPECIAL, NUM_SPECIAL + codebook_size)
    codes = [0] * n
    unfilled = np.ones(n, dtype=bool)

    for t in range(rounds):
        probs = torch.softmax(_logits(model, ids, device)[lo:hi, span].float(), -1)
        if temperature > 0:
            noisy = torch.softmax(torch.log(probs.clamp_min(1e-9)) / temperature, -1)
            picked = torch.multinomial(noisy, 1).squeeze(1)
        else:
            picked = probs.argmax(-1)
        conf = probs.gather(1, picked[:, None]).squeeze(1).cpu().numpy()
        # Cosine schedule: how many positions may stay masked after this round.
        keep_masked = int(n * math.cos(math.pi / 2 * (t + 1) / rounds))
        keep_masked = min(keep_masked, int(unfilled.sum()) - 1)
        conf_now = np.where(unfilled, conf, np.inf)
        commit = np.argsort(-conf_now)
        commit = [i for i in commit if unfilled[i]]
        if keep_masked > 0:
            commit = commit[: len(commit) - keep_masked]
        for i in commit:
            codes[i] = int(picked[i])
            ids[lo + i] = NUM_SPECIAL + codes[i]
            unfilled[i] = False
        if not unfilled.any():
            break
    for i in np.flatnonzero(unfilled):
        codes[int(i)] = int(picked[int(i)])
    return codes


def rigid_between(
    mobile: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rigid move taking ``mobile`` onto ``target``: rotation and centroids.

    Returned separately from :func:`kabsch_onto` because the move sometimes has
    to be applied to coordinates other than the ones it was derived from. The
    pose refiner projects to a rigid transform, so the difference between the
    codes' decode and what is finally written is exactly such a move -- and a
    clash probe judging candidate codes has to apply it to each candidate rather
    than re-running the refiner 256 times.
    """
    mc, tc = mobile - mobile.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(mc.T @ tc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T, mobile.mean(0), target.mean(0)


def apply_rigid(
    coords: np.ndarray, move: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """``coords`` moved by a transform from :func:`rigid_between`."""
    rot, from_c, to_c = move
    return (coords - from_c) @ rot.T + to_c


def kabsch_onto(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigid-body superposition of ``mobile`` onto ``target`` (no scaling)."""
    return apply_rigid(mobile, rigid_between(mobile, target))


def reconcile(
    original: np.ndarray,
    refined: np.ndarray,
    changed: Sequence[int],
    *,
    mode: str = "align",
) -> np.ndarray:
    """Undo the decoder's global reaction to a local code edit.

    ``decode_to_outputs`` runs a transformer over the whole code sequence, so
    changing one code moves every atom: measured over generated ligands, the
    median molecule had *all* of its atoms displaced (median largest move
    1.38 A, q90 4.03) when the intent was to touch 5% of them. The edit itself
    is sound -- the replaced code is right 0.405 of the time against the
    original's 0.093 -- but the molecule that comes back is a different pose,
    and in generation that made things better and worse about equally often
    (49.9% of molecules improved).

    ``align``  superimpose the refined pose onto the original using the atoms
               whose codes did not change, keeping the refined internal
               geometry. Removes the global part only.
    ``splice`` additionally restore the unchanged atoms to their original
               coordinates, so only the edited atoms move. Removes the local
               part too, at the cost of the internal consistency the decoder
               produced.
    ``off``    return the refined pose untouched.
    """
    if mode == "off" or original.shape != refined.shape:
        return refined
    keep = np.ones(len(original), dtype=bool)
    keep[list(changed)] = False
    if keep.sum() < MIN_ANCHOR_ATOMS:
        return refined
    out = kabsch_onto(refined, original) if keep.all() else _align_on(
        refined, original, keep
    )
    if mode == "splice":
        out = out.copy()
        out[keep] = original[keep]
    return out


def _align_on(
    mobile: np.ndarray, target: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    """Superpose using only ``keep`` atoms, then apply to all of them."""
    mc = mobile - mobile[keep].mean(0)
    tc = target - target[keep].mean(0)
    u, _, vt = np.linalg.svd(mc[keep].T @ tc[keep])
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (mc @ rot.T) + target[keep].mean(0)


def load_mlm(ckpt: str, device: torch.device) -> tuple[nn.Module, int]:
    """Load a complex MLM and its ``<mask>`` id."""
    from prolit.model.mlm_module import ProLITMLMModule  # noqa: PLC0415

    module = ProLITMLMModule.load_from_checkpoint(ckpt, map_location=device)
    return module.eval().to(device).model, module.config.model.mask_token_id
