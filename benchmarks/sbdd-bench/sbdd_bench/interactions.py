"""Protein–ligand interaction metrics (category 4) — optional, opt-in.

Vina being low does not mean the *right* interactions are made. With the
``interactions`` dependency group installed (ProLIF + MDAnalysis), this scores
each generated pose against the reference ligand's interaction pattern:

* **Key-residue recovery** — fraction of residues the reference ligand interacts
  with that the generated ligand also contacts. The practically important
  question: does the molecule pick up the residues a known binder uses?
* **Interaction-fingerprint (IFP) Tanimoto** — overlap of the full
  (residue, interaction-type) fingerprints.

Everything degrades gracefully: if ProLIF is not installed the functions return
``None`` so the core pipeline never depends on this module.
"""

from __future__ import annotations

from pathlib import Path


def _available() -> bool:
    try:
        import prolif  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _protein_mol(receptor_pdb: str | Path):
    import prolif as plf
    from rdkit import Chem

    rdmol = Chem.MolFromPDBFile(str(receptor_pdb), removeHs=False, sanitize=False)
    if rdmol is None:
        raise ValueError(f"RDKit could not read receptor {receptor_pdb}")
    return plf.Molecule.from_rdkit(rdmol)


def _ifp_residues(fp, lig_mol, prot_mol) -> set[str]:
    """Run a ProLIF fingerprint for one ligand and return the set of
    "RESID:INTERACTION" keys (e.g. ``"ASP123:HBDonor"``)."""
    import prolif as plf

    plf_lig = plf.Molecule.from_rdkit(lig_mol)
    fp.run_from_iterable([plf_lig], prot_mol, progress=False)
    df = fp.to_dataframe()
    keys: set[str] = set()
    if df.empty:
        return keys
    # columns: (ligand, protein_residue, interaction)
    for col in df.columns:
        if df[col].to_numpy().any():
            _, resid, interaction = col
            keys.add(f"{resid}:{interaction}")
    return keys


def interaction_recovery(receptor_pdb, ref_mol, gen_mols) -> dict[int, dict]:
    """Per generated-mol idx: key-residue recovery + IFP Tanimoto vs reference.

    ``gen_mols`` are :class:`sbdd_bench.molio.GenMol`; only those with a sanitized
    ``mol`` and a 3D conformer are scored. Returns ``{}`` if ProLIF is missing.
    """
    if not _available() or ref_mol is None:
        return {}
    import prolif as plf

    prot = _protein_mol(receptor_pdb)
    fp = plf.Fingerprint()
    try:
        ref_keys = _ifp_residues(fp, ref_mol, prot)
    except Exception:  # noqa: BLE001
        return {}
    ref_residues = {k.split(":")[0] for k in ref_keys}

    out: dict[int, dict] = {}
    for g in gen_mols:
        if g.mol is None or g.mol.GetNumConformers() == 0:
            continue
        try:
            gen_keys = _ifp_residues(fp, g.mol, prot)
        except Exception:  # noqa: BLE001
            continue
        gen_residues = {k.split(":")[0] for k in gen_keys}
        recovery = (
            len(ref_residues & gen_residues) / len(ref_residues)
            if ref_residues else None
        )
        union = ref_keys | gen_keys
        ifp_tanimoto = len(ref_keys & gen_keys) / len(union) if union else None
        out[g.idx] = {
            "key_residue_recovery": recovery,
            "ifp_tanimoto": ifp_tanimoto,
            "n_interactions": len(gen_keys),
        }
    return out
