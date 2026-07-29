"""Machinery shared by the three ProLIT benchmarks.

Each benchmark answers a different question about the same tokenizer --
reconstruction fidelity (plbench), pose rescoring and affinity (ctbench),
pocket-conditioned generation (sbddbench) -- but they must agree on *which*
tokenizer they are talking about, and on how a difference between two methods is
called significant. Those two things live here.
"""
