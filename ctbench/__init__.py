"""complex-tokenizer-bench: evaluation & ablation benchmark for the complex tokenizer.

The trained models (tokenizer VQ-VAE + downstream LM/MLM/heads) live in the
sibling ``pocket-conditioned-ligand-gen`` repo. This package owns everything
else: running inference from those checkpoints, computing task metrics,
statistical comparison against baselines, the joint-vs-single tokenizer ablation,
and the tables/figures.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
