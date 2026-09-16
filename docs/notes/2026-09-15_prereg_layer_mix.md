# Pre-registration: read every layer of the MLM, not just the last one

2026-09-15 ~20:30 JST. Written BEFORE the option is implemented and before any
head using it exists.

## Why this is a new axis

Every affinity head in this project reads **only the final normed hidden state**
of the encoder (`ComplexRescoreModule._encode_states` calls `encoder.encode()`,
which returns `self.norm(x)` after the last of 13 layers). All four readouts
tried so far -- `ligand`, `balanced`, `allatoms`, `concat` -- differ in *which
tokens* are averaged, never in *which layer* is read.

In masked language models the last layer is the one specialised to the masking
objective, and intermediate layers are routinely the better transfer features.
That is untested here, and it is a pure "use the learned representation better"
change: the encoder stays frozen, the corpus is unchanged, and the head keeps
its size.

Not in scope, because the module's own docstring records them as already built
and measured for pose rescoring: learned attention pooling, ligand->pocket
cross-attention, and a GenScore-style sum over ligand-pocket pairs.

## The change

`encode(..., return_all_layers=True)` returns the embedding output plus every
layer's output, each passed through the encoder's existing final LayerNorm so
all 14 states share one scale (no new encoder parameters; the last state is
bit-identical to what `encode()` returns today). The head mixes them with an
ELMo-style scalar mix: `softmax(w)` over 14 learnable weights, initialised to
zeros so the mix starts as the uniform average of all layers.

The 14 weights are created **only when `layer_mix` is on**, so checkpoints
trained without it keep their exact state_dict, and a checkpoint trained with it
fails loudly if loaded without it.

## Arms

* **Treatment** `e250_unionnc_layermix` (NEW, 8 seeds 82-89): corpus
  `union_noconf`, `--corr-weight 0.7`, `--layer-mix`, otherwise the standard
  recipe (`--label-cap 13 --max-epochs 20 --early-stop-patience 4
  --block-size 640`, ligand readout).
* **Control** `e250_unionnc_corr_v3` (8 seeds 58-65): identical corpus, identical
  recipe, no `--layer-mix`. Already trained and scored.

Both families are 8 fresh seeds of the same recipe on the same corpus, so the
only difference is which layers the head reads.

## Predictions (f32 basis, family ensembles of 8)

* **Z1 (scoring).** R(layer-mix 8) - R(control 8) >= **+0.005**.
* **Z2 (ranking).** rho(layer-mix 8) >= rho(control 8) - 0.005.

**Works only if Z1 and Z2 hold.** f16 reported beside.

Reported beside, not part of the verdict: the learned layer weights (which
layers the head actually chose), and the 24-model headline mix with the
control family swapped for this one.

## Cost

8 heads (gpu_1, 1:00) ~= 1.6 pt + 16 inferences ~= 1.6 pt ~= **3.2 pt**, ledger
to about 152 of the 200 pt cap.
