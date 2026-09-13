"""CASF-2016 binding-affinity prediction for the ProLIT affinity head.

Scoring power (Pearson R between predicted and measured pK over all 285 core
complexes) and ranking power (mean within-cluster Spearman rho over the 57
five-ligand clusters), against GenScore, Boltz-2 and Vina under one protocol.

The task is its own benchmark because it is its own paper table and its own
model: the pose head and the affinity head share an architecture and a
tokenizer but not a corpus, not a label, and not a backbone, and each has been
measured to carry none of the other's signal (the pose head scores affinity at
R = -0.036).
"""
