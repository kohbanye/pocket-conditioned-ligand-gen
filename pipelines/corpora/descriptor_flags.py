"""The descriptor-variant flags, in one place for every corpus builder.

Three builders tokenize through :class:`~prolit.tokenizers.pose_encoder.PoseEncoder`
and a fourth fills the descriptor cache directly, and each one has to be told
which descriptor variant the tokenizer it is loading was TRAINED on. Every
variant keeps the descriptor at 33-D, so getting it wrong raises nothing: the
builder hands the tokenizer a well-formed descriptor from a distribution it has
never seen and the corpus looks fine.

That failure has already happened twice. ``pocket_context`` was added to the
cache builder and not to the evaluation path, so the arm was scored on plain
descriptors and reported two kcal of ceiling that was not there; and
``contact_residue_channel`` was added to the cache builder and not to the decoy
builder, which would have silently produced a scoring head trained against the
wrong channel. Declaring the flags once is what stops the third time.

Builders are siblings, so this is imported by bare module name::

    from descriptor_flags import add_descriptor_arguments, descriptor_kwargs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

#: Flag name -> the ``PocketExtractionConfig`` field it sets. Adding a variant
#: means one entry here and nothing in any builder.
DESCRIPTOR_FLAGS: dict[str, str] = {
    "--pocket-context": "pocket_context",
    "--contact-residue-channel": "contact_residue_channel",
}

_HELP: dict[str, str] = {
    "pocket_context": (
        "Widen each LIGAND atom's knn SEARCH set to the pocket, so its empty "
        "neighbour slots hold the nearest protein atoms. Spends two of the four "
        "slots, so the ligand's own local geometry pays for it. Measured to "
        "change nothing at the interface (lDDT-PLI and Contact-F1 tie to three "
        "decimals) and kept as the reported ablation."
    ),
    "contact_residue_channel": (
        "Fill a LIGAND atom's `aa` slot with the residue class it contacts "
        "within 4.5 A, instead of the constant X every ligand row carried. "
        "Costs no descriptor width -- protein rows already use that channel for "
        "their own residue -- and displaces no knn slot."
    ),
}


def add_descriptor_arguments(parser: argparse.ArgumentParser) -> None:
    """Add every descriptor-variant flag, each defaulting to off.

    Off by default is what keeps a builder that does not pass them producing
    byte-identical output to before any of this existed.
    """
    group = parser.add_argument_group(
        "descriptor variant",
        "Must match the cache the --ckpt tokenizer was trained on. The "
        "descriptor is 33-D under every setting, so a mismatch raises nothing.",
    )
    for flag, field in DESCRIPTOR_FLAGS.items():
        group.add_argument(flag, action="store_true", help=_HELP[field])


def descriptor_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The ``PocketExtractionConfig`` keywords those flags imply.

    Reads through :data:`DESCRIPTOR_FLAGS` rather than naming the fields again,
    so a builder cannot pick up the flag and forget to pass it on -- which is
    the shape of both failures in the module docstring.
    """
    out: dict[str, Any] = {}
    for field in DESCRIPTOR_FLAGS.values():
        if hasattr(args, field):
            out[field] = bool(getattr(args, field))
    return out


__all__ = ["DESCRIPTOR_FLAGS", "add_descriptor_arguments", "descriptor_kwargs"]
