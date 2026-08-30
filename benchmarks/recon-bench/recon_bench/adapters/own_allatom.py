"""ProLIT all-atom pocket-ligand tokenizer reconstruction (subprocess).

Replaces the residue-level adapter this bench started with. ProLIT encodes
pocket atoms and ligand atoms with one shared 33-D descriptor, so a single
codebook can cover both, and the ablation question is what that sharing costs
and buys.

One adapter instance = one **arm**. The arms in :data:`ARMS` span the two design
axes the paper argues about:

* **codebook** — one shared book (``joint``) vs a hard partition into a
  protein-only and a ligand-only book (``separate``), 4096 codes each so the two
  arms match on total codebook size and on LM vocabulary and differ only in
  whether the book is shared.
* **frame** — the ligand encoded in the shared pocket frame (placement is in
  every atom token, ``pose_bits=0``) vs in its own canonical frame like a
  single-modality ligand tokenizer (tokens are SE(3)-invariant, so the pose must
  be transmitted separately and ``pose_bits`` prices it).

``binning`` needs no weights at all: it discretizes space on a grid at a
comparable rate and shows what the learned codebook is worth.

Reconstruction runs through ``scripts/own_allatom_reconstruct_cli.py`` in the
source repo's own venv, which dumps one NPZ per complex (per-atom
correspondence, ligand bonds, token counts) that this adapter turns into
:class:`~recon_bench.types.ModalityRecon` rows.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recon_bench import paths
from recon_bench.adapters.base import ReconstructionModel
from recon_bench.types import ModalityRecon, ReconResult, Sample

_CLI = paths.REPO_ROOT / "scripts" / "own_allatom_reconstruct_cli.py"
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


@dataclass
class Arm:
    """One tokenizer configuration to evaluate."""

    name: str
    label: str
    protein_run: str | None = None
    ligand_run: str | None = None
    protein_norm: Path | None = None
    ligand_norm: Path | None = None
    ligand_frame: str = "pocket"
    pose_bits: int | None = None
    kind: str = "vq"
    codebook: str = ""
    # No monitored checkpoint exists for this run -- see best_checkpoint.
    uses_last: bool = False
    #: Repair the decoded ligand geometry with a flow-matching pose refiner
    #: before scoring. Absolute path; the CLI loads it in prolit's venv.
    refiner_ckpt: Path | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _cache(name: str) -> Path:
    return paths.OWN_ALLATOM_CACHE / name


ARMS: dict[str, Arm] = {
    "joint": Arm(
        name="joint",
        label="ProLIT",
        protein_run="xzkjxu9q",
        ligand_run="xzkjxu9q",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
    ),
    "separate": Arm(
        name="separate",
        label="ProLIT (separate tokenizers)",
        protein_run="protein-vqvae-4096",
        ligand_run="ligand-vqvae-4096",
        protein_norm=_cache("normalization_stats_protein.pt"),
        ligand_norm=_cache("normalization_stats_ligand.pt"),
        codebook="2 books (4096+4096)",
        notes="matched to the joint arm on total codebook size and LM vocabulary",
    ),
    "binning": Arm(
        name="binning",
        label="Coordinate binning (no training)",
        kind="binning",
        codebook="grid (10^3 cells x 12 elements)",
        notes="no learned parameters; the rate-matched floor",
    ),
    "joint_e250": Arm(
        name="joint_e250",
        label="ProLIT (250 epochs)",
        protein_run="vq_e250",
        ligand_run="vq_e250",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes=(
            "as joint_dmap, trained 250 epochs instead of 100. Every earlier "
            "run stopped while still improving -- 7 of 8 had their best "
            "checkpoint in the final three epochs -- so the whole arm "
            "comparison was made between undertrained models"
        ),
    ),
    "joint_e250_lig": Arm(
        name="joint_e250_lig",
        label="ProLIT (8.3/8.3 ligand weighting, 250 epochs)",
        protein_run="vq_e250_lig",
        ligand_run="vq_e250_lig",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes=(
            "as joint_dmap, trained 250 epochs instead of 100. Every earlier "
            "run stopped while still improving -- 7 of 8 had their best "
            "checkpoint in the final three epochs -- so the whole arm "
            "comparison was made between undertrained models"
        ),
    ),
    "joint_e250_lig3": Arm(
        name="joint_e250_lig3",
        label="ProLIT (3.0/3.0 ligand weighting, 250 epochs)",
        protein_run="vq_e250_lig3",
        ligand_run="vq_e250_lig3",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes=(
            "as joint_dmap, trained 250 epochs instead of 100. Every earlier "
            "run stopped while still improving -- 7 of 8 had their best "
            "checkpoint in the final three epochs -- so the whole arm "
            "comparison was made between undertrained models. Best of the "
            "three e250 arms: the ligand weight is not a protein/ligand "
            "trade-off at 3.0, which beats BOTH the unweighted arm and 8.3 on "
            "the protein side as well as the ligand side"
        ),
    ),
    "joint_pb3_s7": Arm(
        name="joint_pb3_s7",
        label="ProLIT (protein bonds + long-range, 3.0 ligand weighting)",
        protein_run="vq_pb3_s7",
        ligand_run="vq_pb3_s7",
        uses_last=True,
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes=(
            "joint_e250_lig3 plus bond_distance_all_sources: the 1-2/1-3 terms "
            "stop being ligand-only. Fills the empty cell of the protein-bond "
            "x distance-map square -- joint_pbond_lig had the bonds without "
            "the map and reached a protein 1-2 MAE of 0.160 A against "
            "joint_e250_lig3's 0.198, but paid for it in long-range accuracy "
            "(lDDT 0.936 vs 0.950). lDDT is a local-distance metric and the "
            "protein carried no local term, which is the state the ligand was "
            "in when its bond MAE sat at 0.233"
        ),
    ),
    "joint_pb3_s8": Arm(
        name="joint_pb3_s8",
        label="ProLIT (protein bonds + long-range, 3.0 ligand weighting, seed 8)",
        protein_run="vq_pb3_s8",
        ligand_run="vq_pb3_s8",
        uses_last=True,
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes=(
            "joint_pb3_s7 at a second seed, and the only replicate in this "
            "registry. Without one, the arm ranking has no noise floor: the "
            "ligand weight is non-monotonic on the protein side (3.0 beats "
            "both 0 and 8.3), which is either a balance effect or seed spread, "
            "and nothing measured so far tells the two apart"
        ),
    ),
    "joint_noleak": Arm(
        name="joint_noleak",
        label="ProLIT (leak-free)",
        protein_run="vq_ctrl_p3",
        ligand_run="vq_ctrl_p3",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="CASF-2016 and the sbdd-bench targets held out of training; the published joint weights had 169 of the 285 CASF entries on their train side",
    ),
    "joint_bond": Arm(
        name="joint_bond",
        label="ProLIT (bonded-distance loss)",
        protein_run="vq_bond_p3",
        ligand_run="vq_bond_p3",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="leak-free plus 1-2/1-3 distance losses on the decoded coordinates",
    ),
    "joint_constrained": Arm(
        name="joint_constrained",
        label="ProLIT (constrained chemistry)",
        protein_run="vq_constrained",
        ligand_run="vq_constrained",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="as joint_bond, but the chemistry heads are held to the levels vq_ctrl_p3 reached instead of carrying hand-set weights",
    ),
    "joint_lig_loss": Arm(
        name="joint_lig_loss",
        label="ProLIT (ligand upweighted in the loss)",
        protein_run="vq_lig_loss",
        ligand_run="vq_lig_loss",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="ligand weight 8.3 in the reconstruction loss only; the encoder is pushed, the centroids are left alone",
    ),
    "joint_lig_ema": Arm(
        name="joint_lig_ema",
        label="ProLIT (ligand upweighted in the EMA)",
        protein_run="vq_lig_ema",
        ligand_run="vq_lig_ema",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="ligand weight 8.3 in the codebook EMA only; the centroids are pulled, the encoder is not",
    ),
    "joint_lig_both": Arm(
        name="joint_lig_both",
        label="ProLIT (ligand upweighted in both)",
        protein_run="vq_lig_both",
        ligand_run="vq_lig_both",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="ligand weight 8.3 in loss and EMA, which is how the two were coupled before they were separable",
    ),
    "joint_pbond": Arm(
        name="joint_pbond",
        label="ProLIT (protein bonded-distance loss)",
        protein_run="vq_pbond",
        ligand_run="vq_pbond",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="the 1-2/1-3 terms extended to protein atoms; measured to score 3/8, below the arm without them",
    ),
    "joint_pbond_lig": Arm(
        name="joint_pbond_lig",
        label="ProLIT (protein bonded-distance, ligand-weighted)",
        protein_run="vq_pbond_lig",
        ligand_run="vq_pbond_lig",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="as joint_pbond on top of 8.3/8.3 ligand weighting; the local terms only help when combined with the ligand weight",
    ),
    "joint_dmap": Arm(
        name="joint_dmap",
        label="ProLIT (distance-map loss)",
        protein_run="vq_dmap",
        ligand_run="vq_dmap",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="squared error on every protein-protein distance within 15 A, lDDT's own inclusion radius; the bonded terms call a C-C bond at 1.92 A while backbone lDDT scores CA-CA pairs at ~3.8 A",
    ),
    "joint_dmap_lig": Arm(
        name="joint_dmap_lig",
        label="ProLIT (distance-map loss, ligand-weighted)",
        protein_run="vq_dmap_lig",
        ligand_run="vq_dmap_lig",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="as joint_dmap on top of 8.3/8.3 ligand weighting",
    ),
    "joint_simple": Arm(
        name="joint_simple",
        label="ProLIT (one pairwise term)",
        protein_run="vq_simple",
        ligand_run="vq_simple",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="the four distance terms replaced by one relative-error term over every pair within 15 A; measured to break the ligand (bond MAE 0.075 -> 0.233 A)",
    ),
    "joint_simple_clash": Arm(
        name="joint_simple_clash",
        label="ProLIT (one pairwise term + clash)",
        protein_run="vq_simple_clash",
        ligand_run="vq_simple_clash",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="as joint_simple keeping the clash hinge, which does not recover the bond lengths",
    ),
    "joint_simple_lig": Arm(
        name="joint_simple_lig",
        label="ProLIT (one pairwise term, ligand-weighted)",
        protein_run="vq_simple_lig",
        ligand_run="vq_simple_lig",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="as joint_simple on top of 8.3/8.3 ligand weighting; recovers bond MAE only to 0.150 A",
    ),
    "joint_local": Arm(
        name="joint_local",
        label="ProLIT (local + long-range, no hand weights)",
        protein_run="vq_local",
        ligand_run="vq_local",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="1-2 and 1-3 as one relative-error term and every pair within 15 A as another, both over every atom at weight 1.0; matches joint_dmap on protein but loses the ligand",
    ),
    "joint_local_noclash": Arm(
        name="joint_local_noclash",
        label="ProLIT (local + long-range only)",
        protein_run="vq_local_noclash",
        ligand_run="vq_local_noclash",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="as joint_local with the clash hinge removed, leaving no hand-set weight in the geometry objective",
    ),
    "joint_final": Arm(
        name="joint_final",
        label="ProLIT (one local term, one long-range term)",
        protein_run="vq_final",
        ligand_run="vq_final",
        protein_norm=_cache("normalization_stats.pt"),
        ligand_norm=_cache("normalization_stats.pt"),
        codebook="1 shared book (8192)",
        notes="1-2 and 1-3 together as one relative-error term on ligand rows, every pair within 15 A on protein rows, plus clash; folds bond12 (5.0) and bond13 (2.0) into one term at weight 1.0",
    ),
}

# Ligand-own-frame arms sweep the pose budget instead of fixing one, so the paper
# can report the break-even: how many extra tokens a single-modality ligand
# tokenizer must spend on the rigid transform before its interface metrics match
# a shared-frame tokenizer, which spends none. Fixing a single budget would be
# arbitrary and invites "why not one more token?".
_POSE_SWEEP = [(None, "oracle"), (39, "3tok"), (26, "2tok"), (20, "1.5tok"), (13, "1tok")]
for _bits, _tag in _POSE_SWEEP:
    ARMS[f"localframe_{_tag}"] = Arm(
        name=f"localframe_{_tag}",
        label=f"Ligand-own-frame + {_tag} pose",
        protein_run="protein-vqvae",
        ligand_run="ligand-vqvae-localframe",
        protein_norm=_cache("normalization_stats_protein.pt"),
        ligand_norm=paths.OWN_LOCALFRAME_CACHE / "normalization_stats_ligand.pt",
        ligand_frame="local",
        pose_bits=_bits,
        codebook="2 books (8192+8192)",
        notes="SE(3)-invariant ligand tokens; pose transmitted separately",
    )

# The same two axes again, but on the recipe the joint arm actually uses. The
# arms above cannot answer what sharing buys: ``separate`` and the localframe
# runs all stopped at epoch 98/99 with no distance map and no ligand weighting,
# while ``joint_e250_lig3`` had 250 epochs, the map and a weight of 3.0. Scored
# against those, joint's +0.389 PoseBusters and +0.107 SMILES are the recipe,
# not the codebook -- at matched (old) recipe the sign flips and separate wins
# both, 0.279 vs 0.128 and 0.309 vs 0.104. What survives matching is the
# interface: lDDT-PLI +0.011 and Contact-F1 +0.046, at 76% and 78% per-sample,
# p < 1e-19.
#
# These three runs put the baseline on the same footing: 250 epochs, distance
# map, constrained balancing, held-out eval PDBs, 4096 codes each so the pair
# still matches the joint arm's 8192 and its LM vocabulary. No ligand weighting
# -- that weight exists to share one book between two sources, and there is
# nothing to share here. Per-source masks then switch off what does not apply:
# the protein run scores coord + distance map (1-2/1-3 and clash are ligand-row
# terms and come out zero), the ligand runs score coord + 1-2 + 1-3 + clash (the
# map is a protein-row term and comes out zero). The one asymmetry left is that
# the constraint targets were measured on a joint run, so protein-only chemistry
# clears them easily and releases its multipliers while ligand-only chemistry
# pins them at 1 -- which favours the baseline, the safe direction here.
# The same tokenizer, with the decoded ligand handed to the flow-matching pose
# refiner before anything is scored. Bond perception is where the SMILES goes:
# with reference coordinates it is 99.3%% accurate and with decoded ones 65%%, and
# no tolerance setting recovers more than 70.6%%, so the failure is geometry and
# the refiner is the tool aimed at exactly that.
#
# **This arm's CASP16 numbers are indicative only.** The refiner's corpus is
# drawn from BioLiP2, which carries PDB entries up to 9xim while the CASP16
# ligand targets are from 2024, and the bench indexes those complexes by CASP
# target with no PDB id to exclude them by. Overlap cannot be ruled out, so read
# a gain here as an upper bound and confirm it on a refiner whose corpus is
# bounded by vintage before it becomes a paper number.
ARMS["joint_e250_lig3_refined"] = Arm(
    name="joint_e250_lig3_refined",
    label="ProLIT (3.0 ligand weighting, 250 epochs) + pose refiner",
    protein_run="vq_e250_lig3",
    ligand_run="vq_e250_lig3",
    protein_norm=_cache("normalization_stats.pt"),
    ligand_norm=_cache("normalization_stats.pt"),
    codebook="1 shared book (8192)",
    refiner_ckpt=Path("/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/pocket-ligand-refine/refine_geo_v2/checkpoints/refine-e11-r0.5766.ckpt"),
    notes=(
        "joint_e250_lig3 with the ligand coordinates repaired post-decode. The "
        "pocket handed to the refiner is the REFERENCE one: it is the "
        "conditioning input in both reconstruction and generation, and passing "
        "the decoder's own protein error back in would make the ligand repair "
        "depend on how well the protein happened to come back"
    ),
)

# The Optuna Pareto front's middle point, confirmed at full length. The search
# ran at 15 epochs per trial, and 15-epoch ranking is not 250-epoch ranking --
# at 100 epochs the separate tokenizer beat joint on PoseBusters and SMILES and
# at 250 both reversed -- so the front is a shortlist and this arm is the check.
ARMS["joint_opt5"] = Arm(
    name="joint_opt5",
    label="ProLIT (Optuna front point #5)",
    protein_run="vq_opt5",
    ligand_run="vq_opt5",
    protein_norm=_cache("normalization_stats.pt"),
    ligand_norm=_cache("normalization_stats.pt"),
    codebook="1 shared book (8192)",
    notes=(
        "joint_e250_lig3 with the geometry weights moved to the front's middle "
        "point: bond12 20.63, bond13 6.94, clash 3.56, dmap 2.38, coord held at "
        "1.0. At 15 epochs it bought ligand 1-2 MAE 0.105 -> 0.088 for protein "
        "dmap MAE 0.329 -> 0.378. At 250 epochs its val/atom_coord is 0.1904 "
        "against 0.1021, which the weights predict -- coord's share of the "
        "geometry objective drops from 1/14 to 1/34.5 -- so whether the trade "
        "is worth taking is a question for the benchmark, not the val loss"
    ),
)

ARMS["separate_e250"] = Arm(
    name="separate_e250",
    label="ProLIT (separate tokenizers, 250 epochs)",
    protein_run="vq_sep_prot",
    ligand_run="vq_sep_lig_allatom",
    protein_norm=_cache("normalization_stats_protein.pt"),
    ligand_norm=_cache("normalization_stats_ligand.pt"),
    codebook="2 books (4096+4096)",
    notes=(
        "the codebook axis, recipe-matched to joint_e250_lig3: two private "
        "books in the SHARED pocket frame, so the only difference left is "
        "whether protein and ligand draw from one vocabulary or two"
    ),
)
for _bits, _tag in _POSE_SWEEP:
    ARMS[f"separate_e250_local_{_tag}"] = Arm(
        name=f"separate_e250_local_{_tag}",
        label=f"ProLIT (separate tokenizers, own frame, {_tag} pose)",
        protein_run="vq_sep_prot",
        ligand_run="vq_sep_lig_ligand_localframe",
        protein_norm=_cache("normalization_stats_protein.pt"),
        ligand_norm=paths.OWN_LOCALFRAME_CACHE / "normalization_stats_ligand.pt",
        ligand_frame="local",
        pose_bits=_bits,
        codebook="2 books (4096+4096)",
        notes=(
            "the frame axis, recipe-matched: same two private books as "
            "separate_e250 but the ligand in its own frame, so its tokens are "
            "SE(3)-invariant and the placement has to be paid for separately. "
            "Sweeping the budget is the point -- the shared-frame arms spend "
            "nothing on placement, so the honest question is how many tokens "
            "buy the difference back, not whether one chosen budget wins"
        ),
    )


def _end_to_end(dump) -> dict:
    """The ``e2e_*`` scalars the CLI wrote, as plain columns.

    Empty for dumps made before the CLI scored them, and for the grid baseline,
    whose decoder has no chemistry heads. Empty rather than zero: those arms did
    not fail this test, they did not take it.
    """
    return {
        k[len("e2e_") :]: float(dump[k]) for k in dump.files if k.startswith("e2e_")
    }


def best_checkpoint(
    run: str, min_epoch: int = 90, *, uses_last: bool = False
) -> Path | None:
    """Lowest-``val/atom_coord`` checkpoint of a *finished* run, or None.

    The '/' in the monitored metric name makes every checkpoint its own
    directory, so files land at
    ``<run>/checkpoints/atomvqvae-epoch=NN-val/atom_coord=X.ckpt``.

    ``min_epoch`` refuses checkpoints from a run that is still training. Without
    it a half-trained VQ silently becomes a row in the paper's ablation table.

    Runs trained under multi-GPU DDP before the val stream was unsharded have no
    monitored checkpoints at all -- splitting validation across ranks stopped
    ``val/atom_coord`` from ever being logged, so ``ModelCheckpoint`` never
    fired and only ``last.ckpt`` exists. Such an arm sets ``uses_last=True`` and
    is served ``last.ckpt``. The cost is close to nothing: over the three
    250-epoch arms the last epoch's ``val/atom_coord`` is 0.1% above the best
    epoch's (0.1022 vs 0.1021, 0.1076 vs 0.1075, 0.1204 vs 0.1204), the curve
    being flat by then.

    The flag is per-arm and not a silent fallback, because ``min_epoch`` cannot
    guard it: the epoch lives inside the checkpoint and this package has no
    torch to read it with. Declaring it in the registry puts the judgement
    "this run finished" where someone actually knows the answer.
    """
    ckpt_dir = paths.OWN_VQ_RUNS_DIR / run / "checkpoints"
    found = []
    for path in ckpt_dir.glob("*/atom_coord=*.ckpt"):
        epoch = int(path.parent.name.split("epoch=")[1].split("-")[0])
        if epoch >= min_epoch:
            found.append((float(path.stem.split("=")[-1]), epoch, path))
    if found:
        return min(found)[2]
    last = ckpt_dir / "last.ckpt"
    return last if uses_last and last.exists() else None


class OwnAllAtomAdapter(ReconstructionModel):
    """Reconstruct pocket + ligand with one all-atom tokenizer arm."""

    can_protein = True
    can_ligand = True

    def __init__(
        self,
        arm: str = "joint",
        python: str | None = None,
        out_dir: str | Path | None = None,
        min_epoch: int = 90,
        device: str | None = None,
        **_: object,
    ) -> None:
        if arm not in ARMS:
            raise KeyError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
        self.arm = ARMS[arm]
        self.name = f"own_allatom.{arm}"
        self.python = python or paths.OWN_MODEL_PYTHON
        self.out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_DIR / "own_allatom" / arm
        self.min_epoch = min_epoch
        # None lets the CLI pick; "cpu" is the escape hatch when the GPU is busy
        # (reconstruction is small, so CPU is slow but perfectly workable).
        self.device = device
        self._dumps: dict[str, Path] = {}

    # -- setup -----------------------------------------------------------
    def arm_spec(self) -> dict:
        """Resolve the arm to concrete checkpoint paths for the CLI."""
        spec: dict = {
            "kind": self.arm.kind,
            "ligand_frame": self.arm.ligand_frame,
            "pose_bits": self.arm.pose_bits,
            "refiner_ckpt": (
                str(self.arm.refiner_ckpt) if self.arm.refiner_ckpt else None
            ),
        }
        if self.arm.kind == "binning":
            return spec
        for side in ("protein", "ligand"):
            run = getattr(self.arm, f"{side}_run")
            ckpt = best_checkpoint(
                run, self.min_epoch, uses_last=self.arm.uses_last
            )
            if ckpt is None:
                raise FileNotFoundError(
                    f"arm {self.arm.name!r}: no {side} checkpoint past epoch "
                    f"{self.min_epoch} in {paths.OWN_VQ_RUNS_DIR / run}. "
                    "Still training, or the run name is wrong."
                )
            norm = getattr(self.arm, f"{side}_norm")
            if not norm.exists():
                raise FileNotFoundError(f"arm {self.arm.name!r}: missing {norm}")
            spec[f"{side}_ckpt"] = str(ckpt)
            spec[f"{side}_norm"] = str(norm)
        return spec

    def setup(self) -> None:
        self.arm_spec()

    @classmethod
    def ready_arms(cls, min_epoch: int = 90) -> list[str]:
        """Arms whose weights exist and are past ``min_epoch`` (binning always)."""
        ready = []
        for name in ARMS:
            try:
                cls(arm=name, min_epoch=min_epoch).arm_spec()
            except FileNotFoundError:
                continue
            ready.append(name)
        return ready

    # -- sample-set materialization --------------------------------------
    def materialize(self, samples: list[Sample]) -> list[str]:
        """Run the source repo's CLI over every (protein, ligand) sample."""
        spec = self.arm_spec()
        usable = [s for s in samples if s.protein_pdb and s.ligand_sdf]
        if not usable:
            raise ValueError("all-atom arms need samples with both protein_pdb and ligand_sdf")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "arm.json").write_text(json.dumps(spec, indent=2))
        (self.out_dir / "pairs.json").write_text(
            json.dumps(
                [
                    {
                        "id": s.sample_id,
                        "receptor": str(s.protein_pdb),
                        "ligand": str(s.ligand_sdf),
                    }
                    for s in usable
                ]
            )
        )
        cmd = [
            self.python, str(_CLI),
            "--workdir", str(paths.OWN_MODEL_WORKDIR),
            "--arm", str((self.out_dir / "arm.json").resolve()),
            "--pairs", str((self.out_dir / "pairs.json").resolve()),
            "--out-dir", str(self.out_dir.resolve()),
        ]
        if self.device:
            cmd += ["--device", self.device]
        cmd += ["--receptor-cache", str(paths.RECEPTOR_CACHE.resolve())]
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(paths.OWN_MODEL_WORKDIR),
            capture_output=True,
            text=True,
            check=False,
        )
        self._index_dumps()
        # Dumps persist between runs, so "some NPZ exists" proves nothing: a run
        # that died on the first complex would otherwise be silently scored on
        # whatever a previous, smaller run left behind, and the summary table
        # would look complete. Demand the exit code AND full coverage.
        wanted = {s.sample_id for s in usable}
        missing = wanted - set(self._dumps)
        if proc.returncode != 0 or missing:
            err = (proc.stderr or proc.stdout or "")[-2000:]
            why = []
            if proc.returncode != 0:
                why.append(f"CLI exited {proc.returncode}")
            if missing:
                why.append(f"{len(missing)}/{len(wanted)} complexes have no dump")
            raise RuntimeError(
                f"arm {self.arm.name!r}: reconstruction failed ({'; '.join(why)}). "
                f"Any dumps already in {self.out_dir} are from an earlier run and "
                f"are NOT a valid result for this one.\n{err}"
            )
        return sorted(self._dumps)

    def _index_dumps(self) -> None:
        self._dumps = {p.stem: p for p in sorted(self.out_dir.glob("*.npz"))}

    def dumps(self) -> dict[str, Path]:
        """sample_id -> per-complex NPZ dump, indexing them on first access.

        Exposed because the runner reads ``protein_chain`` / ``protein_resid``
        out of these to define the pocket residue subset that the full-protein
        tokenizers are scored on.
        """
        if not self._dumps:
            self._index_dumps()
        return dict(self._dumps)

    # -- reconstruction interface ----------------------------------------
    def reconstruct(self, sample: Sample) -> ReconResult:
        if not self._dumps:
            self._index_dumps()
        dump = self._dumps.get(sample.sample_id)
        if dump is None:
            return ReconResult(
                self.name, sample.sample_id, ok=False,
                error="sample not materialized; call materialize() first",
            )
        return self._parse_dump(sample.sample_id, dump)

    def _parse_dump(self, sample_id: str, dump: Path) -> ReconResult:
        d = np.load(dump, allow_pickle=False)
        prot_ref, prot_rec = d["protein_ref"], d["protein_rec"]
        lig_ref, lig_rec = d["ligand_ref"], d["ligand_rec"]
        prot_elements = [str(e) for e in d["protein_elements"]]
        lig_elements = [str(e) for e in d["ligand_elements"]]
        atom_names = [str(n) for n in d["protein_atom_names"]]
        res_keys_all = [
            (str(c), int(r)) for c, r in zip(d["protein_chain"], d["protein_resid"], strict=True)
        ]
        rate = {
            "bits_protein": float(d["bits_protein"]),
            "bits_ligand": float(d["bits_ligand"]),
            "pose_bits": float(d["pose_bits"]),
            "arm_label": self.arm.label,
            "arm_codebook": self.arm.codebook,
            "ligand_frame": self.arm.ligand_frame,
        }

        modalities: list[ModalityRecon] = []
        # CA-only view so this arm lines up with ESM3 / FoldToken, which
        # reconstruct backbones and nothing else.
        ca = [i for i, n in enumerate(atom_names) if n.strip() == "CA"]
        if ca:
            modalities.append(
                ModalityRecon(
                    modality="protein_backbone",
                    ref=prot_ref[ca], rec=prot_rec[ca], atom_kind="CA",
                    n_residues=len(ca),
                    n_tokens=int(d["n_tokens_protein"]),
                    res_keys=[res_keys_all[i] for i in ca],
                    extra=dict(rate),
                )
            )
        # All-atom view: what this tokenizer actually reconstructs, and the only
        # scope on which side-chain geometry at the interface is visible.
        bb = [i for i, n in enumerate(atom_names) if n.strip() in _BACKBONE_ATOMS]
        modalities.append(
            ModalityRecon(
                modality="protein_allatom",
                ref=prot_ref, rec=prot_rec, atom_kind="heavy",
                n_residues=len({k for k in res_keys_all}),
                n_tokens=int(d["n_tokens_protein"]),
                res_keys=res_keys_all,
                extra={**rate, "n_backbone_atoms": len(bb)},
            )
        )
        modalities.append(
            ModalityRecon(
                modality="ligand",
                ref=lig_ref, rec=lig_rec, atom_kind="heavy",
                n_tokens=int(d["n_tokens_ligand"]),
                extra={
                    **rate,
                    "elements": lig_elements,
                    "bonds": [(int(a), int(b)) for a, b, _ in d["ligand_bonds"]],
                    "bond_orders": [int(o) for *_, o in d["ligand_bonds"]],
                    # Everything scored against the reference bonds measures
                    # geometry alone -- it gives the model back the molecule it
                    # was meant to recover. These are the end-to-end answer.
                    "end_to_end": _end_to_end(d),
                },
            )
        )
        # Protein and ligand stacked in the frame they were reconstructed in --
        # no per-modality superposition, so this is where a lost binding pose
        # actually shows up.
        modalities.append(
            ModalityRecon(
                modality="complex",
                ref=np.vstack([prot_ref, lig_ref]),
                rec=np.vstack([prot_rec, lig_rec]),
                atom_kind="heavy",
                n_tokens=int(d["n_tokens_protein"]) + int(d["n_tokens_ligand"]),
                extra={
                    **rate,
                    "n_protein_rows": int(prot_ref.shape[0]),
                    "protein_elements": prot_elements,
                    "ligand_elements": lig_elements,
                },
            )
        )
        return ReconResult(self.name, sample_id, modalities=modalities)
