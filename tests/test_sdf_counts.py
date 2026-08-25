"""The V2000 counts line is fixed-width, and reading it with ``split()`` is silent.

A ligand with 98 atoms and 101 bonds writes " 98101" -- the three-character
fields touch. ``split()`` then reports 98101 atoms, and the parser walks the
bond block, the properties, and whatever follows as if they were atoms, at
coordinates that look perfectly plausible. Nothing raises. Five CASF-2016
targets are written that way, and every one of them scored against a pocket
extracted around a ligand three times its real size.
"""

from prolit.tokenizers.ligand import parse_sdf_text


def _sdf(counts: str, n_atoms: int, n_bonds: int) -> str:
    head = "mol\n  prog\n\n" + counts + "\n"
    atoms = "".join(
        f"{i:10.4f}{0.0:10.4f}{0.0:10.4f} C   0  0  0  0  0\n"
        for i in range(n_atoms)
    )
    bonds = "".join(f"{1:3d}{2:3d}{1:3d}{0:3d}\n" for _ in range(n_bonds))
    return head + atoms + bonds + "M  END\n$$$$\n"


def test_three_digit_bond_count_is_not_swallowed() -> None:
    """98 atoms / 101 bonds -> " 98101", the case that broke the five targets."""
    mols = parse_sdf_text(_sdf(" 98101  0  0  0  0  0  0  0  0999 V2000", 98, 101))
    assert len(mols) == 1
    assert len(mols[0]["atoms"]) == 98
    assert len(mols[0]["bonds"]) == 101


def test_three_digit_atom_count() -> None:
    mols = parse_sdf_text(_sdf("140142  0  0  0  0  0  0  0  0999 V2000", 140, 142))
    assert len(mols[0]["atoms"]) == 140
    assert len(mols[0]["bonds"]) == 142


def test_ordinary_two_digit_counts_unchanged() -> None:
    mols = parse_sdf_text(_sdf(" 26 25  0  0  0  0  0  0  0  0999 V2000", 26, 25))
    assert len(mols[0]["atoms"]) == 26
    assert len(mols[0]["bonds"]) == 25


def test_writer_that_ignores_the_column_widths() -> None:
    """Space-separated counts still parse -- the fallback path."""
    mols = parse_sdf_text(_sdf("9 8 0 0 0 0 0 0 0 0999 V2000", 9, 8))
    assert len(mols[0]["atoms"]) == 9
    assert len(mols[0]["bonds"]) == 8
