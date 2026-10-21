# Affinity improvement: independent primary-source literature audit

Date: 2026-09-13. Scope: literature and experimental design only; no historical project result notes were read. This is a targeted evidence review, not an exhaustive systematic review. Published primary papers, their author repositories and identifiable preprints were searched using GenScore, CleanSplit, LP-PDBBind, DataSAIL, interaction pretraining, ProFSA and BindNet queries. Search-engine publication-age metadata was not treated as a publication date. Some publisher pages blocked full-page fetching; searchable publisher text, author-hosted papers and repositories were used where available. Numerical results below are reported results, not independently reproduced results.

## Main judgment

The strongest literature-backed direction for a tokenizer paper is to improve how the tokenizer and LM preserve **cross-interface interactions**, then demonstrate the gain using a fixed, small prediction head. A larger head could mask representational damage. More pretraining examples alone are not sufficient evidence of progress: the pretraining corpus, tokenizer-fitting corpus, label sources and baseline training corpus all need overlap accounting.

“Beat GenScore” needs an explicit protocol: matching IDs, structure preparation, supervised labels, external pretraining allowances, checkpoint selection and ensemble size. Published CASF correlations across incompatible training regimes do not answer this question.

## Verified source ledger

### 1. GenScore: the original method and the comparison target

Shen et al. (2023), *A generalized protein–ligand scoring framework with balanced scoring, docking, ranking and screening powers*, **Chemical Science 14, 8129–8146**, DOI [10.1039/D3SC02044D](https://pubs.rsc.org/en/content/articlelanding/2023/sc/d3sc02044d). [Full article](https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02044d), [author code](https://github.com/sc8668/GenScore).

Verified: GenScore learns residue–atom distance likelihoods using a mixture density network, with an adjustable affinity-correlation objective. Its model variants trade scoring/ranking against docking/screening. Therefore a zero-affinity-weight checkpoint and an affinity-specialized checkpoint are different comparators. The repository includes GT and GatedGCN variants and atom/residue contribution modes. Its example pocket generation uses a 10 Å cutoff.

Implication (our inference): borrow distance/contact supervision **during representation learning**, not necessarily its MDN as the downstream head. Compare the exact affinity-oriented variant and preserve its full training/checkpoint recipe. “GenScore” alone is an insufficient baseline specification. Some secondary web records incorrectly list this paper as JCIM; the RSC DOI above is authoritative.

### 2. CleanSplit/GEMS: direct evidence that protocol changes the gap

Graber et al. (2025), *Resolving data bias improves generalization in binding affinity prediction*, **Nature Machine Intelligence**, DOI [10.1038/s42256-025-01124-5](https://www.nature.com/articles/s42256-025-01124-5). [Author code and split files](https://github.com/camlab-ethz/GEMS), [author-institution PDF](https://www.research-collection.ethz.ch/bitstreams/926650e3-980c-49bf-8562-b8f9514606f5/download).

Verified: on CASF2016 (285 complexes), CleanSplit-trained GenScore reports Pearson 0.780/RMSE 1.362; GEMS 0.803/1.308. GEMS combines interaction graphs and language-model embeddings. CASF2016 has 57 target clusters of five complexes; the paper also evaluates within-cluster rankings. The filtering examines ligand similarity, protein structure and aligned ligand placement, and **uses affinity differences to preserve activity cliffs**. Thus CleanSplit is not simply a protein-sequence cutoff, nor a label-independent filtering protocol.

Implication (our inference): use the released split for historical comparability, but add an independently locked, input-only split for the strongest no-leak claim. Audit the exact filtering code before reimplementing thresholds: summary articles can misstate their Boolean logic. Protein-only, ligand-only and nearest-neighbour baselines are essential controls for whether interaction representations help.

### 3. LP-PDBBind: a distinct and version-sensitive benchmark

Li et al., *Leak Proof PDBBind: A Reorganized Dataset of Protein-Ligand Complexes for More Generalizable Binding Affinity Prediction*. [arXiv v3, dated 2026-01-10](https://arxiv.org/html/2308.09639v3); initial preprint 2023.

Verified v3 protocol: training versus validation/test protein sequence similarity ≤0.5; validation versus test ≤0.9; ligand Morgan-fingerprint **Dice** similarity ≤0.99 between splits. Protein similarity uses aligned sequence matches. CL1 excludes covalent binders, underrepresented elements and steric clashes; CL2 additionally removes IC50 records. Reported experiments train on CL1 and test on CL2. BDB2020+ adds post-2020 structures/BindingDB labels with similarity filtering.

Implication (our inference): do not call LP-PDBBind a strict cold-scaffold benchmark; 0.99 Dice mostly targets near identity. Do not substitute Tanimoto thresholds for Dice. Record actual release, cleaning tier and ID lists: “CL1/CL2/CL3” are data cleaning labels, not three guaranteed novelty levels. Temporal splitting alone also allows related ligands and targets across the boundary.

### 4. DataSAIL: match the intended generalization question

Joeres et al. (2025), *Data splitting to avoid information leakage with DataSAIL*, **Nature Communications 16, 3337**, DOI [10.1038/s41467-025-58606-8](https://www.nature.com/articles/s41467-025-58606-8). [Split-evaluation API](https://datasail.readthedocs.io/en/latest/interfaces/dl_eval.html).

Verified: S1 controls similarity along one entity axis; S2 controls both ligand and protein axes. S2 can discard interactions where the two entities are assigned incompatible partitions. It optimizes similarity-induced leakage, not a universal guarantee of zero overlap. The authors explicitly warn that OOD splits can be overly pessimistic for an application that deliberately predicts close analogues.

Implication (our inference): publish both a lead-optimization regime (new ligands for known target families) and a novel-target/novel-ligand regime. Fix similarity definitions, retained IDs and discarded counts. These measure different intended uses; collapsing them into one score hides a scientifically important distinction.

### 5. ProFSA: unusually direct precedent for simple-head representation evaluation

Gao et al. (2024), *Self-supervised Pocket Pretraining via Protein Fragment-Surroundings Alignment*, **ICLR 2024**. [Conference paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/6cc31b44d88dce8380d36e81485cd07f-Paper-Conference.pdf), [author repository](https://github.com/bowen-gao/ProFSA).

Verified: drug-like fragments extracted from proteins provide over five million pseudo pocket–ligand pairs. Contrastive pretraining aligns pocket representations with pretrained pseudo-ligand representations. Affinity experiments use **frozen encoders plus an MLP on concatenated representations**. On ATOM3D/PDBBind v2019 30% protein identity split, ProFSA reports Pearson 0.628 versus its Uni-Mol baseline 0.558; at 60%, 0.764 versus 0.645. These are not CASF/CleanSplit numbers, and the split does not by itself establish ligand novelty.

Implication (our inference): compare interaction-focused versus independent-component pretraining under the same tokenizer/LM/head. A frozen-feature result is an especially clear argument that the representation improved. Pseudo-complex generation remains a later option because its parent protein structures require overlap filtering and its chemistry differs from small-molecule ligands.

### 6. BindNet: self-supervision should involve the interface

Feng et al., *Protein-ligand binding representation learning from fine-grained interactions*, **ICLR 2024**. [Author preprint](https://arxiv.org/abs/2311.16160), [author project](https://atomlab.yanyanlan.com/project/bindnet/).

Verified: a transformer interaction module is pretrained with atomic pair-distance prediction and masked ligand reconstruction. The framework explicitly models protein–ligand interactions rather than pretraining two independent components alone.

Implication (our inference): add masked cross-interface reconstruction/contact or distance targets to the **LM pretraining loss**. Remove auxiliary decoders during affinity inference. Hold downstream pooling and head fixed. This isolates interaction information learned by tokens/LM without making the affinity head responsible for rebuilding all geometry.

### 7. Uni-Mol: useful representation baseline, with a substantial external-data allowance

[Official Uni-Mol README](https://github.com/deepmodeling/Uni-Mol/blob/main/unimol/README.md).

Verified: the repository describes molecular pretraining on 209 million 3D conformations and pocket pretraining on three million candidate pockets. Its binding task includes pose prediction; that is not automatically an affinity benchmark result.

Implication (our inference): separate “same permitted pretraining corpus” experiments from “best available external pretrained model” experiments. External pretraining can be scientifically legitimate, but uncontrolled structural overlap prevents a strict unseen-complex claim. Transfer weights cannot simply be called leak-free because their training lacked affinity labels.

### 8. BA-Pred: a newer comparator and an ensemble caveat

Sim and Lee (2026), *BA-Pred and RMSD-Pred: Integrated Graph Neural Network Models for Accurate Protein–Ligand Binding Affinity and Binding Pose Prediction*, **JCIM 66, 3480–3495**, DOI [10.1021/acs.jcim.5c02591](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02591). [Author code](https://github.com/eightmm/BA-Pred).

Verified: the paper reports CleanSplit/CASF Pearson 0.818 and RMSE 1.293 with five-fold CV ensembling; a 25-model ensemble reaches 0.823/1.282. The architecture includes separate protein, ligand and interaction modules. Affinity pooling uses ligand nodes. The repository’s packaged default checkpoints are **random-split** models, not evidence of a CleanSplit-trained baseline.

Implication (our inference): GenScore is not the only relevant current comparator. However, compare single models separately from equal-size ensembles and retrain on common IDs where necessary. Ligand-token pooling is a low-complexity candidate, but this paper does not prove it is the reason for the performance gain.

### 9. Emerging leakage critique: keep it as a preprint, not settled verdict

Mattsson and Walters (2026), *Identifying and Addressing Systematic Data Leakage in Protein-Ligand Affinity Benchmarks*, [bioRxiv 10.64898/2026.06.29.735309](https://www.biorxiv.org/content/10.64898/2026.06.29.735309v1.full).

Verified searchable primary text: ligand-only baselines reach mean Pearson 0.66 on FEP+4 and 0.36 on OpenFE under the studied protocol. Novelty tiers use maximum Morgan Tanimoto to training/validation compounds. The authors explicitly state this does not establish that Boltz-2/IsoDDE obtain most of their performance by memorization. Their ligand-only models include target one-hot identity; the label “ligand-only” does not mean absence of target identity.

Implication (our inference): report target-identity availability in every ablation. A ligand-only baseline with a target ID and one without are different controls. More recent large affinity models belong in a separate external-data track unless their supervised corpora can be reconciled with the common evaluation split.

## Actionable experiments: recommendations, not literature-proven outcomes

These are hypotheses for the local model; confirmation requires local representation and error audits.

| Priority | Mechanism | Fixed elements | Diagnostic success condition |
|---|---|---|---|
| 1 | Quantization fidelity audit: atom/contact labels, charge, aromaticity, H-bond donor/acceptor identity, short-range distances | Existing LM/head | Token-conditioned probes preserve interface chemistry beyond occupancy-only baselines; find collision pairs with different contact patterns |
| 1 | Compare chemical features recoverable from continuous pre-tokenizer embeddings versus discrete codes | Probe family and data split | Large continuous-to-discrete loss identifies a tokenizer bottleneck; both weak implies encoder/input bottleneck |
| 1 | Increase interface importance in tokenizer reconstruction, rather than globally increasing vocabulary | Token budget, LM compute, downstream head | Better contact reconstruction and held-out affinity/ranking together; distinguish physical signal from a larger model |
| 2 | Cross-interface masking, contact/distance reconstruction in LM pretraining | Corpus, LM size, affinity head | Frozen linear/small-MLP affinity improves; masked-only language loss need not be lowest |
| 2 | Multi-view consistency for atom ordering, equivalent serializations and small physically plausible coordinate perturbations | Head and effective example budget | Same-complex prediction variance decreases without erasing stereochemical or meaningful pose distinctions |
| 2 | Ligand-only, interface-only, and separated ligand/pocket mean pooling | Same LM embeddings and two-layer-or-smaller head | Improvement reproduces across fixed seeds without adding an interaction network to the head |
| 3 | Train-only within-target pairwise affinity loss, combined with calibrated regression | Input, head and supervised IDs | Improves macro within-target ranking, not only pooled Pearson; restrict to compatible assay/label types |
| 3 | Pseudo pocket–fragment interaction pretraining | Downstream protocol and head | Adds value after parent-protein similarity exclusion; gains survive dataset-size/compute-matched controls |

For the first rows, visualize nearest neighbours in token/code space, especially **different physical interfaces sharing the same code pattern** and **nearly identical interfaces producing unstable codes**. Codebook utilization/perplexity alone cannot demonstrate chemically useful information preservation. If quantization is involved, the decisive evidence is a downstream-relevant distortion curve at matched token rate, not merely reconstruction RMSD.

A minimal attribution grid is: current/new tokenizer × current/interface-pretrained LM, each with frozen linear probe and the same small MLP. Include random-init LM and continuous-embedding upper-control where feasible. Use equal token/parameter/compute budgets or clearly state which varies; a shorter tokenization changes effective context and compute, so identical epoch counts are not always fair.

## Benchmark contract to lock before further model selection

1. One ID manifest per split, one label table with measurement type/units, and one structure-preparation specification. Mark all preprocessing failures and report coverage; never silently compare each model on its own successful subset. Give both common-intersection metrics and failures on the full intended test set.
2. Audit four stages independently: tokenizer fitting, LM pretraining, affinity training and validation/model selection. Audit PDB ID, standardized ligand identity, protein sequence/family, pocket structure, interaction similarity and label-source duplicates. “No direct affinity-label overlap” is weaker than “no complex exposure.”
3. Publish pooled Pearson and RMSE together with macro target Spearman, target-centred error/correlation where defined, and ligand/protein novelty strata. Pearson can improve via between-target offsets without better ligand ranking. Report target sample counts and affinity ranges; avoid interpreting five-point correlations as stable mechanistic facts.
4. Compare paired predictions on identical samples. Use target-cluster bootstrap for confidence intervals of correlation/error differences and report multiple random seeds. One favourable seed or a point estimate delta is insufficient to claim a win.
5. If CASF or another test set has already been repeatedly inspected for improvement, classify it as an exploratory benchmark. Diagnose failure mechanisms there, validate changes on development folds, and reserve a separate locked final set for confirmation. Do not delete hard examples after viewing residuals.
6. Keep pretrained and training-from-scratch tracks distinct; specify baseline variant, source revision, checkpoint, ensemble size, tuning budget and selection metric. Add strong simple ligand/protein/nearest-neighbour controls so a tokenizer claim cannot rest on dataset shortcuts.

## What this review does not establish

It does not establish the cause of the local model’s errors or promise a numerical gain. It does not independently reproduce any external score, certify any existing checkpoint’s training manifest, or prove any cited split has no leakage under every definition. Novelty claims for a particular tokenizer design require a dedicated representation/tokenization search after its actual design is inspected. The highest-value immediate result is an evidence-based choice of *which information is lost*, followed by a small, attributable experiment on locked development data.
