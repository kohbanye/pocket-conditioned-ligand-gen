"""Descriptor caches, token streams, and the datasets built on them.

Kept empty of re-exports on purpose: every module here pulls in torch (and
several pull in RDKit), so re-exporting them would make ``import src.data``
expensive for callers that only want one dataset class. Import directly:

    from src.data.atom_descriptors import AtomComplexDescriptorDataModule
    from src.data.lm_dataset import LMTokenDataModule
    from src.data.rescore_dataset import RescoreDataModule
"""
