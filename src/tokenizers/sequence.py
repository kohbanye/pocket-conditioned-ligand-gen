"""Token sequence assembler for the <p>...<s>...<l>... format."""

from __future__ import annotations

import re


class TokenSequenceAssembler:
    """Assemble and disassemble the 3-section token sequence format.

    Output format::

        <p>G_20 A_10 L_4 A_5 V_33</p><s>MKTIIALSYIF...</s><l>C_20 O_23 N_6</l>

    - ``<p>...</p>``: Pocket residues as ``AA_structcode`` (space-separated)
    - ``<s>...</s>``: Full protein sequence (contiguous 1-letter AAs)
    - ``<l>...</l>``: Ligand atoms as ``element_structcode`` (space-separated)
    """

    _PATTERN = re.compile(r"<p>(.*?)</p><s>(.*?)</s><l>(.*?)</l>", re.DOTALL)

    def assemble(
        self,
        pocket_tokens: list[str],
        full_sequence: str,
        ligand_tokens: list[str],
    ) -> str:
        """Assemble components into the token sequence text format.

        Args:
            pocket_tokens: Pocket residue tokens, e.g. ``["G_20", "A_10"]``.
            full_sequence: Full protein 1-letter sequence, e.g. ``"MKTII..."``.
            ligand_tokens: Ligand atom tokens, e.g. ``["C_20", "O_23"]``.

        Returns:
            Assembled token sequence string.
        """
        pocket_str = " ".join(pocket_tokens)
        ligand_str = " ".join(ligand_tokens)
        return f"<p>{pocket_str}</p><s>{full_sequence}</s><l>{ligand_str}</l>"

    def disassemble(self, text: str) -> dict[str, str | list[str]]:
        """Parse a token sequence string back into components.

        Returns:
            Dict with keys ``pocket_tokens``, ``full_sequence``,
            ``ligand_tokens``.
        """
        match = self._PATTERN.match(text)
        if not match:
            msg = f"Invalid token sequence format: {text[:80]!r}"
            raise ValueError(msg)

        pocket_str, full_sequence, ligand_str = match.groups()

        return {
            "pocket_tokens": pocket_str.split() if pocket_str.strip() else [],
            "full_sequence": full_sequence,
            "ligand_tokens": ligand_str.split() if ligand_str.strip() else [],
        }
