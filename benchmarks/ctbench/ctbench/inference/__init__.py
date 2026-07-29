"""Inference drivers: load ProLIT's trained models and produce dumps.

These modules import the model library (``prolit.tokenizers.*``,
``prolit.model.*``, ``prolit.data.*``) -- never the ``scripts/*`` eval layer --
load a variant's checkpoints, run each task, and write per-sample dumps in the
:mod:`ctbench.io_dumps` schema. They require torch + a GPU and are meant to run
under qsub, not in unit tests.

``prolit`` is a normal workspace dependency, so no import-path juggling is
needed here.
"""

from __future__ import annotations
