"""ProLIT: a chemistry-aware all-atom tokenizer for 3D protein-ligand interfaces.

A Transformer VQ-VAE maps the coordinates and chemistry of a pocket and its
ligand to one sequence of discrete interface tokens, and reconstructs the
complex from them. Two language models are trained on those tokens: a masked
encoder (ProLIT-MLM) used for binding-pose rescoring, and a causal decoder
(ProLIT-CLM) used for pocket-conditioned 3D ligand generation.

Start from :mod:`prolit.api`, which is the surface the benchmarks are written
against. The subpackages behind it are:

``prolit.chem``        RDKit / PDB parsing, pocket extraction, geometry
``prolit.tokenizers``  the descriptor schema, the VQ-VAE, the token vocabulary
``prolit.data``        descriptor caches, token streams, datasets
``prolit.model``       the VQ-VAE, CLM, MLM, scoring heads and the pose refiner
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
