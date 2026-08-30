# Why generation cannot mask clashing tokens (2026-08-19)

The generated ligands drive atoms through the receptor wall: closest
ligand-protein heavy-atom contact has a median of 1.73 A where the crystal
ligands' median is 2.74 A, and only 12% of molecules are clash-free by the
benchmark's 0.75-vdW rule (crystal ligands: 85%). The refiner lifts that to
25%, sampling temperature 0.6 to 31%. Neither is close.

An attractive fix: the receptor is *given* at generation time, so tokens that
would place an atom inside it are inadmissible and could simply be masked out
of the LM's logits -- no retraining, no new parameters, the same argument that
justifies solving bond orders and relaxing local geometry at decode time.

**It does not work, because a code does not have a position.** The VQ decoder
is a transformer over the whole token sequence, so the coordinate a given code
decodes to depends on the other tokens and on its slot. Measured over 40 random
codes, each placed in 8 random contexts at 3 sequence positions:

    same code, varied context/slot: mean deviation from its own mean position
                                    1.383 A  (max 2.710 A)
    median spacing between different codes' positions
                                    1.456 A

The context-dependence is as large as the distance between distinct codes.
There is therefore no "position of token k" to test against the receptor before
committing to it, and any precomputed mask would be wrong by about the same
amount as the quantity it is trying to resolve.

Two consequences worth keeping:

* Constrained decoding against the receptor is off the table for this decoder.
  It would only become possible with a decoder whose per-atom output is a
  function of that atom's code alone.
* The clash is not a tokenizer defect. Encoding and decoding the *reference*
  ligands round-trips at 0.386 A RMSD and 80% clash-free, against the crystal
  ligands' own 85% -- the representation holds a clash-free pose fine. What
  fails is the LM's choice of tokens.
