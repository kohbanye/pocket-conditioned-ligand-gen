"""The supported surface of ProLIT: load a tokenizer, encode, decode, sample.

Everything a benchmark, a notebook, or an external user should need is
re-exported here, so callers do not have to know which private module a helper
happens to live in today. Anything not listed in ``__all__`` is internal and may
move without notice.

Two tokenizer arms are supported and expose the same encode/decode surface, so
downstream code can hold either behind one name:

* **joint** -- one VQ-VAE over a shared codebook, covering pocket atoms and
  ligand atoms alike. This is ProLIT proper. Load with :func:`load_tokenizer`
  passing a single checkpoint.
* **separate** -- two single-modality VQ-VAEs (protein-only, ligand-only)
  stitched into one contiguous code space; the paper's ablation. Load with
  :func:`load_separate_tokenizer`.

A minimal round trip::

    from prolit.api import (
        load_tokenizer, load_norm_stats, pocket_atoms_from_pdb_text,
        LigandAtomDescriptor, AtomLMVocab,
    )

    vq = load_tokenizer(ckpt, codebook_size=8192, device=device)
    norm = load_norm_stats(stats_path, device)
    pocket, frame = pocket_atoms_from_pdb_text(receptor_text, ligand_coords)
    vocab = AtomLMVocab(codebook_size=8192)

Checkpoints and normalization statistics always travel together: a VQ-VAE
decodes into the descriptor space its ``normalization_stats.pt`` defines, and
pairing a checkpoint with the wrong statistics produces silently wrong
coordinates rather than an error.
"""

from __future__ import annotations

from prolit.chem.bond_orders import (
    assign_bond_orders,
    mol_from_decoded,
    target_bond_sums,
)
from prolit.chem.mol2 import mol_to_dict, parse_mol2_multi
from prolit.chem.pdb_io import (
    infer_bonds,
    read_heavy_atoms,
    write_full_protein_pdb,
)
from prolit.chem.rigid_fit import rigid_pocket_fit, vdw_radii
from prolit.chem.torsion_fit import torsion_pocket_fit
from prolit.data.rescore_dataset import ligand_mask
from prolit.tokenizers.atom import (
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    atom_descriptor_to_coords,
    precompute_receptor_atom_features,
    precompute_receptor_atom_features_from_text,
    rotate_atom_descriptor,
)
from prolit.tokenizers.descriptor_schema import (
    ATOM_DESCRIPTOR_DIM,
    ATOM_LAYOUT,
    LIGAND_ELEMENT_VOCAB,
    SOURCE_LIGAND_IDX,
    SOURCE_PROTEIN_IDX,
    fields_by_name,
)
from prolit.tokenizers.geometry import random_rotation_matrix
from prolit.tokenizers.ligand import (
    parse_ligand_pdb_text,
    parse_sdf,
    parse_sdf_text,
)
from prolit.tokenizers.lm_vocab import AtomLMVocab
from prolit.tokenizers.loaders import (
    load_causal_lm,
    load_masked_lm,
    load_norm_stats,
    load_pose_refiner,
    load_scoring_head,
    load_separate_tokenizer,
    load_tokenizer,
)
from prolit.tokenizers.pose_encoder import PoseEncoder
from prolit.tokenizers.protein import (
    PocketAtomData,
    compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates,
    precompute_pocket_atom_candidates_from_text,
)

__all__ = [
    "ATOM_DESCRIPTOR_DIM",
    "ATOM_LAYOUT",
    "LIGAND_ELEMENT_VOCAB",
    "SOURCE_LIGAND_IDX",
    "SOURCE_PROTEIN_IDX",
    "AtomLMVocab",
    "LigandAtomDescriptor",
    "PocketAtomData",
    "PoseEncoder",
    "ProteinAtomDescriptor",
    "assign_bond_orders",
    "atom_descriptor_to_coords",
    "compute_canonical_frame",
    "extract_pocket_atoms_from_candidates",
    "fields_by_name",
    "infer_bonds",
    "ligand_mask",
    "load_causal_lm",
    "load_masked_lm",
    "load_norm_stats",
    "load_pose_refiner",
    "load_scoring_head",
    "load_separate_tokenizer",
    "load_tokenizer",
    "mol_from_decoded",
    "mol_to_dict",
    "parse_ligand_pdb_text",
    "parse_mol2_multi",
    "parse_sdf",
    "parse_sdf_text",
    "precompute_pocket_atom_candidates",
    "precompute_pocket_atom_candidates_from_text",
    "precompute_receptor_atom_features",
    "precompute_receptor_atom_features_from_text",
    "random_rotation_matrix",
    "read_heavy_atoms",
    "rigid_pocket_fit",
    "rotate_atom_descriptor",
    "target_bond_sums",
    "torsion_pocket_fit",
    "vdw_radii",
    "write_full_protein_pdb",
]
