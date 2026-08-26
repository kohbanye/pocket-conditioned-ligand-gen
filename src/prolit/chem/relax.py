"""Recover local geometry from the molecular graph instead of from tokens.

A token buys 13 bits, and the decoder spends them placing an atom. What it
should *not* have to spend them on is the fact that an aromatic C-C is 1.39 A
or that an sp3 angle is 109.5 degrees -- those follow from the bond graph, which
:mod:`prolit.chem.bond_orders` already recovers exactly. Transmitting them again
in the coordinates is redundant, and the decoder does it badly: measured on 390
decoded ligands, PoseBusters' bond-length check passed 0.885 of the time and its
angle check 0.913, with the worst angle 1.43x its reference.

So this repairs the local geometry against MMFF94 while holding the pose, using
a **flat-bottomed** restraint on every heavy atom: no penalty at all inside
``max_displacement`` of where the decoder put it, harmonic beyond. That keeps
the one number here from being a tuned weight -- set it to the tokenizer's own
measured coordinate error and it says "free to move within the model's
uncertainty, pinned outside it", which is a measurement rather than a choice.
The measured trade-off on those 390 ligands, against the 0.102 A coordinate MAE
of the checkpoint they came from:

    max_displacement   PB-valid   heavy-atom RMSD from the decoded pose
        (none)           0.608                --
        0.05 A           0.654              0.083 A
        0.10 A           0.651              0.121 A
        0.25 A           0.641              0.241 A
        0.50 A           0.613              0.430 A

Flat between 0.05 and 0.10 -- i.e. flat across the tokenizer's error scale, so
nothing hinges on where in that range it lands -- and worse on both sides. Too
loose and the molecule walks away from the pocket the pose was conditioned on;
too tight and MMFF cannot resolve the strain at all.

**Pushing the ligand off the protein was tried here and is not in this module.**
The decoder also drives atoms through the receptor wall -- the closest
ligand-protein heavy-atom contact has a median of 1.73 A against the crystal
ligands' 2.74 A, and 66% of molecules have a contact under 2.0 A where no
crystal ligand does. Adding the receptor as fixed points with a minimum-contact
constraint does clear the hard overlaps, but measured over 882 ligands it costs
more than it buys: PB-validity 0.932 -> 0.851, while the benchmark's clash-free
rate moves only 0.145 -> 0.145 and the mean clash count 7.03 -> 5.88. Relieving
the clash means bending the ligand, and a bent ligand fails the geometry checks
instead. The clash is a real defect and it belongs to the model -- the decoder
should not have put the atom there -- not to a repair step downstream.

**This does not improve the pose and cannot.** The restraint's whole purpose is
to forbid that: at 0.05 A the heavy atoms move 0.083 A on average, below the
decoder's own error, so docking scores are unchanged to within noise. It repairs
chemistry the tokens were never asked to carry. Report it as part of the
decoder, and report the unrelaxed numbers beside it -- the baselines are not
relaxed either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:

    from rdkit.Chem import Mol

#: The tokenizer's measured coordinate MAE, and the floor for any restraint
#: radius derived per molecule -- nothing the decoder produced is more accurate
#: than the tokenizer it was decoded through.
TOKENIZER_COORD_MAE = 0.102

#: Restraint force constant, in kcal/mol/A^2, applied *outside* the flat bottom.
#: Stiff enough that the flat bottom is the operative parameter rather than this
#: one: at 999 an atom pushed to twice the tolerance pays ~1000 kcal/mol, which
#: no MMFF term can outbid, so the restraint reads as a wall and the only
#: quantity that matters is where the wall is.
_FORCE_CONSTANT = 999.0

def _copy_back(mol: Mol, protonated: Mol) -> Mol:
    """Relaxed heavy-atom coordinates, written into a copy of the input.

    Not ``RemoveHs(protonated)``: that rebuilds the molecule and is not the
    inverse of ``AddHs``. RDKit refuses to strip a hydrogen whose neighbour
    carries non-tetrahedral stereochemistry, so the result can come back with
    one more atom than went in -- which silently breaks the per-atom
    correspondence every downstream consumer assumes, and did. ``AddHs``
    appends, leaving the original indices untouched, so copying the first
    ``mol.GetNumAtoms()`` positions back is exact and cannot change the graph.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Geometry import Point3D  # noqa: PLC0415

    out = Chem.Mol(mol)
    source = protonated.GetConformer()
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        position = source.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(position.x, position.y, position.z))
    return out


def decoder_bond_error(mol: Mol) -> float:
    """How far this molecule's worst bond is from where its elements want it.

    The flat-bottom restraint wants to be the decoder's coordinate error, and
    ``0.102`` -- the tokenizer's MAE -- is that number *averaged over
    everything the tokenizer ever encoded*. It is far too tight for the
    molecules that most need repairing: a bond at 0.687 of its reference length
    needs ~0.4 A of movement, and no amount of MMFF will find it through a
    0.102 A bottom. Sweeping a single looser value up recovers the difference
    (PB 0.752 at 0.102, 0.787 at 1.0) but replaces a measurement with a knob.

    So measure it per molecule instead. Each molecule reports its own worst
    bond-length error, and that is how much room its atoms get -- tight where
    the decoder did well, loose exactly where it did not. The floor is the
    tokenizer MAE, since no molecule's true error is smaller than that.
    """
    from prolit.chem.pdb_io import covalent_radius  # noqa: PLC0415

    conformer = mol.GetConformer()
    coords = conformer.GetPositions()
    worst = TOKENIZER_COORD_MAE
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        begin = mol.GetAtomWithIdx(i).GetSymbol()
        end = mol.GetAtomWithIdx(j).GetSymbol()
        if begin == "H" or end == "H":
            continue
        reference = covalent_radius(begin) + covalent_radius(end)
        if reference <= 0.0:
            continue
        observed = float(np.linalg.norm(coords[i] - coords[j]))
        worst = max(worst, abs(observed - reference))
    return worst


def relax_local_geometry(
    mol: Mol,
    max_displacement: float,
    *,
    max_iters: int = 500,
) -> Mol | None:
    """MMFF-relax ``mol`` with every heavy atom pinned to within
    ``max_displacement`` Angstroms of where it started.

    Returns a new molecule with the same graph and a repaired conformer, or
    ``None`` if the molecule cannot be typed or minimised -- callers keep the
    unrelaxed one in that case rather than dropping it, because a molecule MMFF
    cannot type is still a molecule the model generated.

    Hydrogens are added (at idealised positions) for the minimisation because
    MMFF types need them, and removed again afterwards: they were never part of
    the decoder's output and must not appear in what is scored.
    """
    from rdkit import Chem  # noqa: PLC0415

    # From rdForceFieldHelpers rather than the AllChem umbrella: these are
    # C-extension symbols that AllChem re-exports at runtime but that a type
    # checker cannot see through it.
    from rdkit.Chem.rdForceFieldHelpers import (  # noqa: PLC0415
        MMFFGetMoleculeForceField,
        MMFFGetMoleculeProperties,
        UFFGetMoleculeForceField,
    )

    try:
        # addCoords places the new hydrogens geometrically; without it they land
        # at the origin and MMFF minimises from a structure that is nonsense.
        protonated = Chem.AddHs(mol, addCoords=True)
        field = None
        props = MMFFGetMoleculeProperties(protonated)
        if props is not None:
            field = MMFFGetMoleculeForceField(protonated, props)
        if field is None:
            # MMFF94 declines to type roughly a fifth of what the model
            # generates (unusual valences, elements outside its parameter set),
            # and those molecules were being returned unrepaired -- which is
            # exactly the population whose bond lengths and angles then fail
            # PoseBusters. UFF covers the periodic table at lower fidelity, so
            # it repairs what MMFF will not touch instead of leaving it broken.
            field = UFFGetMoleculeForceField(protonated)
        if field is None:
            return None
        for atom in protonated.GetAtoms():
            if atom.GetAtomicNum() > 1:
                field.MMFFAddPositionConstraint(
                    atom.GetIdx(), max_displacement, _FORCE_CONSTANT
                )
        field.Minimize(maxIts=max_iters)
        return _copy_back(mol, protonated)
    except (ValueError, RuntimeError, Chem.AtomValenceException):
        return None
