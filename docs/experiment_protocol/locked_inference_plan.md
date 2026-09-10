# Locked inference plan (phases 3–4.2a)

Plan lock date: 31 August 2026.

Status: families and coding locked **before** p-values were computed.

Input: already-archived locked-test summaries and `test_evaluation.json` (confusion matrix / per-class recall). No new FL training. No test selection after looking at p-values.

Basis: `docs/research_protocol_20260824.md` §8 and RQ1–RQ3 in the manuscript.

## Pairing unit

- Seed list: `101, 211, 307, 401, 503`.
- Algorithms and clean/attack conditions on the same partition are paired by seed.
- Where a family requires it, the three locked pairs (apple, cherry, potato) **enter the same seed block**: they are averaged inside the seed, so confirmatory n = 5 (amendment 8 September 2026). The old n = 15 (pairs treated as exchangeable) is not confirmatory.

## Tests

Primary test: paired permutation test on the mean difference (all 2^n sign assignments when n ≤ 15).

Sensitivity: Wilcoxon signed-rank on the same differences.

Interval: percentile bootstrap 95% CI of the mean paired difference, 10,000 replicates, RNG seed 20260824.

Holm correction inside each family below, not across families.

With n = 5 seeds the exact two-sided permutation p cannot be smaller than 2/32 = 0.0625. That is not a reason to enlarge the family or to change the α level after the fact.

## Family F1 — clean utility (RQ1), E = 1

Metric: macro-F1 on the locked test.

| ID | Comparison | n |
|----|------------|---|
| F1.1 | FedProx − FedAvg, α = 0.1 | 5 seeds |
| F1.2 | FedProx − FedAvg, α = 0.5 | 5 |
| F1.3 | FedProx − FedAvg, IID | 5 |

Holm across F1.1–F1.3.

## Family F1b — heterogeneity (RQ1), FedAvg E = 1

Same seed, different partitions (pairing by the RNG list, not by an identical image layout).

| ID | Comparison | n |
|----|------------|---|
| F1b.1 | α = 0.1 − IID, macro-F1 | 5 |
| F1b.2 | α = 0.5 − IID, macro-F1 | 5 |

Holm across F1b.1–F1b.2.

## Family F2a — source harm (RQ2), FedAvg flip 1.0

| ID | H0 | n |
|----|----|---|
| F2a.1 | mean source-recall harm = 0, α = 0.1 | 5 seeds (3 pairs averaged within seed) |
| F2a.2 | same, α = 0.5 | 5 |
| F2a.3 | same, IID | 5 |

Holm across F2a.1–F2a.3.

## Family F2b — visibility gap (RQ2), FedAvg flip 1.0

Difference: source-recall harm minus accuracy drop, same scale.

| ID | H0 | n |
|----|----|---|
| F2b.1 | mean(harm − acc_drop) = 0, α = 0.1 | 5 seeds (3 pairs averaged within seed) |
| F2b.2 | same, α = 0.5 | 5 |
| F2b.3 | same, IID | 5 |

Holm across F2b.1–F2b.3.

## Family F2c — FedProx is not a defence (RQ2), flip 1.0

| ID | Comparison | n |
|----|------------|---|
| F2c.1 | FedProx − FedAvg, source harm, α = 0.1 | 5 seeds (3 pairs averaged within seed) |
| F2c.2 | same, α = 0.5 | 5 |
| F2c.3 | same, IID | 5 |

Holm across F2c.1–F2c.3.

## Family F3 — robust aggregators at α = 0.1 (RQ3)

Confirmatory only at α = 0.1 (the manuscript headline). Other α remain descriptive.

| ID | Comparison | n |
|----|------------|---|
| F3.1 | median − attacked FedAvg, source-recall recovery, α = 0.1 | 5 seeds (3 pairs averaged within seed) |
| F3.2 | trimmed mean − attacked FedAvg, same | 5 |
| F3.3 | MultiKrum − attacked FedAvg, same | 5 |
| F3.4 | Krum − attacked FedAvg, same | 5 |
| F3.5 | clean Krum − clean FedAvg, macro-F1, α = 0.1 | 5 seeds |

Holm across F3.1–F3.5.

## Mixed model (RQ2)

Protocol formula:

`class_recall ~ monopoly + entropy + algorithm + attack + monopoly:attack + algorithm:attack + (1 | seed) + (1 | class)`

**Coding of `attack` (documented deviation).** The indicator is at class-within-job, not job-level: `targeted = 1` only if that class is the locked label-flip source in that job. A job-level `attack` would mix targeted harm with collateral on the other 18 classes and would not answer RQ2.

Coverage: phase 3 clean E = 1 FedAvg/FedProx (30 jobs) and phase 4.1 flip 1.0 (90 jobs). All 19 classes. Outcome: recall on the locked test from archived `test_evaluation.json`. `monopoly` and `normalized_entropy` from the train partition.

The Gaussian LMM is the confirmatory model from the protocol. Binomial GEE on (correct, support) clustered by seed is sensitivity, because per-image predictions exist in the laboratory archive (class aggregation gives the same binomial count as Bernoulli per image).

Exploratory (not Holm, not confirmatory): the same LMM plus `log(test support)`.

Spearman ρ of monopoly vs harm on 45 FedAvg jobs remains a descriptive point with a seed-clustered bootstrap CI; it is not a third p-family (jobs inside a seed are not independent).

## What is not in this plan

- E = 5 under attack (not run).
- Choosing extra comparisons after p-values.
- Phase 5 PV-19-full as a new test family (confirmation stays descriptive; the slice is smaller and not a full pair).

## Amendment — 8 September 2026 (inferential unit)

**Status:** dated amendment after isolated manuscript reviews. It does **not** add or drop comparisons, change Holm families, or use new training jobs. It corrects the exchangeability assumption for families that previously listed n = 15.

**What each seed controls.** The integers `101, 211, 307, 401, 503` draw the client partition from the locked train pool and are also passed to each client as `FL_SEED` (`seed_everything`: Python, NumPy, PyTorch). Attack JSON files use a separate flip-mask seed (`1337`); eligible source examples still depend on that partition. A confirmatory seed is therefore a joint replicate of ownership pattern and training randomness.

**Confirmatory unit for F2a, F2b, F2c, F3.1–F3.4.** Average the three locked pairs (apple, cherry, potato) inside each seed, then run the exact sign-flip and the percentile bootstrap on the five seed-level means (n = 5). That is equivalent to flipping all three pair gaps inside a seed together. The mean of the five seed-means equals the unweighted 15-job mean; only the p-value, bootstrap interval, and Wilcoxon on the clustered units change.

**What this implies.** The two-sided permutation floor is 0.0625 for those families as well. They cannot reject at 0.05. A bootstrap interval that excludes zero when all five seed-means have the same sign does not rescue the exact test.

**Diagnostic only.** The old n = 15 sign-flip (pairs treated as independently flippable) is stored in `locked_inference_results.json` under `legacy_job_level` and is not a confirmatory claim.

**Script:** `docs/manuscript/locked_inference.py`.

## Amendment — 8 September 2026 (α vs monopoly and dose qMc)

**Status:** secondary analysis of archived phase-4.1 and 4.2b jobs. Not a new Holm family. Table 10 is unchanged.

**Question.** The abstract asked whether monopoly explains harm better than α. Table 10 has no α term. Under the provisioned-owner rule, flip fraction q implies expected global corrupted share qMc.

**What was computed.** (1) Flip 1.0, n=90: Spearman Mc–harm pooled and within α; partial Spearman given α; leave-one-seed-out RMSE of harm ~ α vs α+Mc. (2) Fractions 0.25/0.50/1.0, n=270: harm vs qMc and vs realized flipped/N_c; OLS/LOSO with qMc and Mc together.

**Result used in the manuscript.** At flip 1.0, realized flipped/N_c equals Mc. Adding Mc to α improves seed-blocked prediction of harm, but the within-α=0.1 association is weak. Across fractions, harm tracks qMc; Mc given qMc is compatible with zero. No dose-matched extra experiment.

## Amendment — 8 September 2026 (LMM specification audit)

**Status:** secondary. Table 10 protocol formula and coefficients are unchanged.

**What was computed.** Named variance components; residual/fitted summaries; Pearson/VIF for Mc vs normalised entropy; monopoly-only and entropy-only LMMs; protocol formula with (1|job); protocol plus job-level `attack_job` alongside targeting; paired source-recall harm ~ Mc + α + algorithm + pair with (1|seed).

**Result used in the manuscript.** Class intercept is the boundary component. Monopoly×targeted survives job intercept (−0.692) and attack_job (−0.720); attack_job ≈ 0. Monopoly main effect is not interpretable (sign flips without entropy). Interaction is three source classes. Paired-harm Mc = 0.85 aligns with the Table 11 dose reading. GEE does not validate Table 10.

## Amendment — 9 September 2026 (preregistration / sealed / confirmatory wording)

**Status:** wording and chronology only. No new Holm family, no new training, no change to Table 9 p-values or Table 10 coefficients.

**What this plan is.** Dataset, partitions, pairs, and slices were locked on 24 August 2026 (`docs/research_protocol_20260824.md`) before phase 3–5 training. This inference document was written on 31 August 2026 on archived `test_evaluation.json` files and slice summaries, before p-values were computed. That is a locked analysis of archived predictions. It is **not** a public preregistration (OSF or similar). Blindness to means, plots, or per-class recalls is **not** documented.

**Sealed test.** The 850-image capped test is scored once per completed matrix slice and reused on later slices. It is not a once-only holdout kept unseen until the whole study ended.

**Confirmatory.** Family IDs F1–F3 are comparison bundles in this document. PV-19-full is a locked follow-on on more images of the same 19 classes. Validation learning-curve summaries exist in job histories; reporting them would be exploratory, not forbidden by a preregistration rule.

**Manuscript.** Table 13 lists dates, artefacts (`dataset_id` and summary JSON paths), and what was already on disk. Laboratory git SHAs are omitted because they were not part of the August lock record. The public GitHub tag is a September archive of this already-scored protocol.

## Amendment — 9 September 2026 (own-baseline source harm)

**Status:** secondary metric on archived 4.2a and 4.2c jobs. Not a Holm family. Table 9 and Table 10 unchanged.

**Question.** Table 4 source harm for aggregator \(a\) is versus clean FedAvg, so it mixes clean source-class cost with attack-induced damage.

**What was computed.** For each locked source class, clean source recall on the matching clean-robust job minus attacked source recall on the 4.2a job (same seed, α, aggregator). Pair-averaged like Table 4. Identity check: harm vs clean FedAvg = clean source cost + own-baseline harm.

**Result used in the manuscript.** At α = 0.1, Krum own-baseline harm is 0.250 (not zero); median / trimmed mean remain 0.536 / 0.559. Table 7 reports both columns. Figure 5 uses own-baseline harm on the vertical axis.

## Amendment — 10 September 2026 (visibility-gap arithmetic)

**Status:** secondary. Not a Holm family. Table 9 F2b formula unchanged; its interpretation is operational.

**Question.** \(H_s-\Delta\mathrm{Accuracy}=0\) compares a class-level change with a prevalence-weighted change. Report \(\pi_s\), \(\pi_s H_s\), collateral \(\Delta\mathrm{Accuracy}-\pi_s H_s\), and \(P(\hat y=\mathrm{target}\mid y=\mathrm{source})\).

**Result used in the manuscript.** Locked test supports: Apple 50, Cherry 46, Potato 24 of 850. At α = 0.1 FedAvg, \(\pi_s H_s\) is 0.035 / 0.034 / 0.016 against accuracy drops 0.016 / 0.042 / 0.004. Pair-mean source term 0.028 vs drop 0.021. Source-to-target rates 0.584 / 0.691 / 0.425; Potato errors are more dispersed. No detector was evaluated.

## Amendment — 10 September 2026 (executable specification)

**Status:** methods dump. Not a Holm family. No new training. No new table.

**Question.** Sections 3 and 8 must specify Dirichlet allocation, leaf-group integrity among clients, leaf-ID origin and the 99% rule, scored-run optimiser and freeze, BN buffers, attack-mask timing, aggregation tensors and Krum \(n>2f+2\), and what a public artefact would contain.

**Result used in the manuscript.** Group-safe Dirichlet (class shares across clients; groups not split across silos; min 20 images/client). Scored freeze is `features` only; AdamW \(10^{-3}\), batch 16, BN running stats still aggregated. Flip mask seed 1337 drawn once; fractions nested. Flower 1.32.1. No public OSF/DOI claimed.

## Amendment — 10 September 2026 (public GitHub artefact)

**Status:** submission archive. Not a Holm family. Not a public preregistration. No new training.

**What was posted.** https://github.com/Aleksandar91/smart-agri-fl tag `pv19-protocol-v2` (commit `78f68f301a7d9d0372258edd2b7930b898d3e91f`). English-only dump of the PV-19 protocol (scored `fl-client` / `fl-server` import graph, launchers, attack configs for the three locked pairs, dataset and partition manifests, attacker map, job registries, slim `test_evaluation.json`, `locked_inference.py`, figures 1–5).

**Not in the dump.** Raw PlantVillage images; `*.npz` checkpoints (`checkpoint_sha256` is in each slim evaluation); per-image `predictions`; `fl_history.json`; Raspberry Pi / mTLS / DP analysis scripts; the pre-24-August PlantVillage-v2 campaign; Serbian laboratory notes.
