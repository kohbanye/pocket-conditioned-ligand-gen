"""In-house pocket-conditioned-ligand-gen adapter (Ours).

Drives the model's own ``scripts/generate_ligands_for_target.py`` in its working
copy's uv venv. That script extracts the pocket around the reference ligand,
encodes it with the protein VQ-VAE, autoregressively samples ligand codes from
the LM, decodes them to 3D in the real pocket frame, and writes ``generated.sdf``
straight into our output directory (the reference ligand, written by the script
as the ``ref`` entry, is dropped by the evaluator so every model is scored on
generated molecules only).

Tokenizer mode / checkpoints are selected via constructor kwargs OR environment
variables (so ``run_generation.py --models own`` can drive any variant without a
code change — the pose_rescoring_bench driver just sets the env before the subprocess):

    SBDD_OWN_MODE            legacy | allatom | separate   (default: legacy)
    SBDD_OWN_LM_CKPT         LM checkpoint (abs path)
    SBDD_OWN_VQVAE_CKPT      VQ-VAE ckpt: legacy 2-codebook OR all-atom VQ
    SBDD_OWN_CODEBOOK_SIZE   codebook size (allatom: combined 8192;
                             separate: PER-MODALITY, e.g. 4096)
    SBDD_OWN_REFINE_CKPT     optional pose-refiner ckpt
    SBDD_OWN_NORM_STATS      optional all-atom normalization_stats.pt
    SBDD_OWN_SEP_PROTEIN_CKPT / _PROTEIN_NORM / _LIGAND_CKPT / _LIGAND_NORM
                             separate-arm protein/ligand VQ ckpts + norm stats
    SBDD_OWN_CACHE_DIR       legacy-path descriptor cache dir
"""

from __future__ import annotations

import os
from pathlib import Path

from sbdd_bench import paths
from sbdd_bench.adapters.base import GenerativeModel
from sbdd_bench.types import GenResult, Target


def _envp(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v) if v else None


class OwnAdapter(GenerativeModel):
    name = "own"
    needs_pocket_pdb = False

    def __init__(self, lm_ckpt=None, vqvae_ckpt=None, cache_dir=None, python=None,
                 batch_size: int = 128, temperature: float = 1.0, top_p: float = 0.95,
                 mode: str | None = None, codebook_size: int | None = None,
                 refine_ckpt=None, norm_stats=None,
                 sep_protein_ckpt=None, sep_protein_norm=None,
                 sep_ligand_ckpt=None, sep_ligand_norm=None, **_):
        self.mode = (mode or os.environ.get("SBDD_OWN_MODE", "legacy")).lower()
        self.lm_ckpt = Path(lm_ckpt) if lm_ckpt else (_envp("SBDD_OWN_LM_CKPT") or paths.OWN_LM_CKPT)
        self.vqvae_ckpt = (Path(vqvae_ckpt) if vqvae_ckpt
                           else (_envp("SBDD_OWN_VQVAE_CKPT") or paths.OWN_VQVAE_CKPT))
        self.cache_dir = (Path(cache_dir) if cache_dir
                          else (_envp("SBDD_OWN_CACHE_DIR") or paths.OWN_DESCRIPTOR_CACHE))
        self.python = python or paths.OWN_PYTHON
        self.workdir = paths.OWN_MODEL_WORKDIR
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        cb = codebook_size or os.environ.get("SBDD_OWN_CODEBOOK_SIZE")
        self.codebook_size = int(cb) if cb else None
        self.refine_ckpt = Path(refine_ckpt) if refine_ckpt else _envp("SBDD_OWN_REFINE_CKPT")
        # Sampling seed. The generator seeds torch with --seed (default 0), so
        # a second run with the same arguments reproduces the SAME molecules --
        # oversampled pools must vary this or they just duplicate pool 1.
        self.seed = int(os.environ.get("SBDD_OWN_SEED", "0"))
        # Sampling knobs the driver may override without a code change. The
        # constructor defaults are kept so existing callers are unaffected.
        self.temperature = float(os.environ.get("SBDD_OWN_TEMPERATURE", temperature))
        self.top_p = float(os.environ.get("SBDD_OWN_TOP_P", top_p))
        self.min_atoms_frac = float(os.environ.get("SBDD_OWN_MIN_ATOMS_FRAC", "0"))
        self.min_atoms_abs = int(os.environ.get("SBDD_OWN_MIN_ATOMS_ABS", "0"))
        self.norm_stats = Path(norm_stats) if norm_stats else _envp("SBDD_OWN_NORM_STATS")
        self.sep_protein_ckpt = (Path(sep_protein_ckpt) if sep_protein_ckpt
                                 else _envp("SBDD_OWN_SEP_PROTEIN_CKPT"))
        self.sep_protein_norm = (Path(sep_protein_norm) if sep_protein_norm
                                 else _envp("SBDD_OWN_SEP_PROTEIN_NORM"))
        self.sep_ligand_ckpt = (Path(sep_ligand_ckpt) if sep_ligand_ckpt
                                else _envp("SBDD_OWN_SEP_LIGAND_CKPT"))
        self.sep_ligand_norm = (Path(sep_ligand_norm) if sep_ligand_norm
                                else _envp("SBDD_OWN_SEP_LIGAND_NORM"))

    def setup(self) -> None:
        checks = [(self.lm_ckpt, "LM checkpoint")]
        if self.mode == "separate":
            checks += [
                (self.sep_protein_ckpt, "separate protein VQ ckpt"),
                (self.sep_protein_norm, "separate protein norm"),
                (self.sep_ligand_ckpt, "separate ligand VQ ckpt"),
                (self.sep_ligand_norm, "separate ligand norm"),
            ]
        else:
            checks.append((self.vqvae_ckpt, "VQ-VAE checkpoint"))
        for p, what in checks:
            if p is None or not Path(p).exists():
                raise FileNotFoundError(
                    f"own {what} missing: {p}. Set the SBDD_OWN_* env vars / "
                    "run scripts/fetch_weights.py --own."
                )

    def _mode_args(self) -> list[str]:
        args: list[str] = []
        if self.mode == "separate":
            args += [
                "--separate-protein-ckpt", str(Path(self.sep_protein_ckpt).resolve()),
                "--separate-protein-norm", str(Path(self.sep_protein_norm).resolve()),
                "--separate-ligand-ckpt", str(Path(self.sep_ligand_ckpt).resolve()),
                "--separate-ligand-norm", str(Path(self.sep_ligand_norm).resolve()),
            ]
        elif self.mode == "allatom":
            args += ["--all-atom", "--vqvae-ckpt", str(self.vqvae_ckpt.resolve())]
        else:  # legacy
            args += ["--vqvae-ckpt", str(self.vqvae_ckpt.resolve())]
        if self.codebook_size is not None:
            args += ["--codebook-size", str(self.codebook_size)]
        if self.norm_stats is not None:
            args += ["--norm-stats", str(Path(self.norm_stats).resolve())]
        if self.refine_ckpt is not None:
            args += ["--refine-ckpt", str(Path(self.refine_ckpt).resolve())]
        return args

    def generate(self, target: Target, n_samples: int, out_dir: Path) -> GenResult:
        self.setup()
        script = self.workdir / "scripts" / "generate_ligands_for_target.py"
        env = dict(os.environ, PYTHONPATH=str(self.workdir))
        cmd = [
            self.python, script,
            "--receptor", target.receptor_pdb,
            "--ref-ligand", target.ref_ligand_sdf,
            "--lm-ckpt", self.lm_ckpt.resolve(),
            "--cache-dir", self.cache_dir.resolve(),
            "--out-dir", out_dir.resolve(),
            "--num-samples", n_samples,
            "--batch-size", self.batch_size,
            "--temperature", self.temperature,
            "--top-p", self.top_p,
            "--seed", self.seed,
            "--min-atoms-frac", self.min_atoms_frac,
            "--min-atoms-abs", self.min_atoms_abs,
            *self._mode_args(),
        ]
        proc = self._run(cmd, cwd=self.workdir, env=env)
        sdf = out_dir / "generated.sdf"
        if not sdf.exists():
            return GenResult(
                self.name, target.target_id, ok=False, n_requested=n_samples,
                error=(proc.stderr or proc.stdout or "")[-2000:],
            )
        from rdkit import Chem

        n = sum(
            1 for m in Chem.SDMolSupplier(str(sdf), sanitize=False)
            if m is not None and (not m.HasProp("_Name") or not m.GetProp("_Name").startswith("ref"))
        )
        return GenResult(self.name, target.target_id, sdf=sdf,
                         n_requested=n_samples, n_generated=n)
