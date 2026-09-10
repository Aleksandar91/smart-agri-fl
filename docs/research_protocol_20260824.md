# Locked evaluation protocol (PV-19 class-concentrated FL)

Start date: 24 August 2026.

This is the operational lock for the PV-19-capped scoring slices (phases 3–4.2c) and the PV-19-full follow-on (phase 5). It is not a public preregistration and not a DOI.

Primary aim: measure how class-concentrated non-IID partitions affect utility, targeted label flipping, and robust aggregation in a reproducible five-silo Flower testbed.

This document is the protocol. Any change to a pre-specified decision must be recorded under “Deviations” before the analysis that uses that change.

Raspberry Pi, mTLS, differential privacy, field photographs, and the earlier PlantVillage-v2 / PV-27 engineering campaign are **out of scope for this artefact**. They are not part of the paper’s claims.

## 1. Scope

The contribution is not a new FL optimiser and not a field plant-disease detector. PlantVillage is a controlled image task used to:

1. measure class-level heterogeneity, not only a global Dirichlet concentration;
2. relate class ownership concentration to clean utility and source-class harm;
3. compare aggregators on identical partitions.

## 2. Research questions

**RQ1.** How do Dirichlet heterogeneity and class-ownership concentration affect global macro-F1, balanced accuracy, worst-class recall, and per-client utility at a matched optimisation budget?

**RQ2.** How far does class ownership concentration predict targeted label-flip harm and the gap between overall and source-class metrics?

**RQ3.** How do FedAvg, FedProx, and selected robust aggregators change the clean-utility vs attack-robustness trade-off on identical partitions?

## 3. Datasets

### 3.1 Primary benchmark: PV-19-capped

- Source: original PlantVillage RGB (`raw/color`).
- Classes: 19 classes with at least 99% class-aware `leaf_id` coverage; after the class-level filter, remaining images without a leaf ID are dropped.
- Cap: at most 300 images per class (whole leaf groups; groups are not split to hit 300 exactly).
- Size before the train/validation/test split: 5,526 images in 993 physical leaf IDs.
- `dataset_id`: `pv19-capped-62b5b2119fb2`.
- Role: the full multi-seed utility, heterogeneity, and attack/defense matrix.

The earlier laboratory subset `infra/fl-data-pv-v2` is **not** the locked benchmark.

### 3.2 Confirmatory benchmark: PV-19-full

- Same source, same 19 classes and canonical names as PV-19-capped.
- No 300-image cap.
- `dataset_id`: `pv19-full-774007483a1d` (20,597 images).
- Role: a pre-specified subset of key configurations, not a repeat of the full matrix.

An exploratory PV-27-capped audit (1,200 images without a known leaf ID; unresolved cross-split perceptual pairs) is **not** used for confirmatory claims. That audit is why the primary set is PV-19.

## 4. Group-safe split

One locked global split is made before FL partitioning:

- train pool: 70%;
- development validation: 15%;
- final test: 15%.

Percentages are applied approximately per class at the **group** level, not per image. All images of the same physical leaf belong to one split. The final test is not used to choose hyperparameters, rounds, aggregators, attack pairs, or checkpoints.

Group-ID hierarchy:

1. class-aware PlantVillage `leaf_id`, when available;
2. identical SHA-256 image content;
3. a confirmed near-duplicate / perceptual group;
4. a unique image ID only when the previous links are absent.

Perceptual candidates are not merged automatically from a loose threshold. The audit reports threshold, distance, component size, and representative pairs.

A scan of the local full RGB tree found 54,305 RGB files in 38 classes and 40,328 keys in `leaf-map.json`. Coverage is complete for most but not all classes. Presence of `leaf-map.json` is not enough to call a split group-safe.

## 5. Dataset manifest and gates

Each variant archives: source name/version; relative path, canonical class, file size, SHA-256; group ID and its origin; split; subset and split seeds; class list and class-to-index map; counts per class and split; exact-duplicate and perceptual-audit reports; no group-ID or SHA-256 intersection across splits; hash of the manifest itself and the script version.

The dataset gate passes only if:

1. no known group crosses a split boundary;
2. no SHA-256 crosses a split boundary;
3. every class has train samples and, where group counts allow, validation and test;
4. all manifest paths exist;
5. a rerun with the same parameters yields the same manifest hash;
6. unknown or incomplete `leaf_id` coverage is quantified.

## 6. FL partitions and seeds

Client partitions are drawn only from the locked train pool. Validation and test images are never written into client training manifests.

Primary setting:

- five clients;
- alpha 0.1, 0.5, and an IID control;
- FedAvg and FedProx;
- local epochs 1 and 5 on the clean matrix;
- equal round count and a pre-specified local-work budget;
- five paired partition/training seeds: `101, 211, 307, 401, 503`.

Locked communication budget for the primary clean matrix: **10 rounds**. E=1 and E=5 are compared at that same round count. FedProx uses `μ = 0.01` unless a separate μ sweep is locked.

Dataset subset and global split use a separate fixed seed and do not change across these runs. The same partition seed is used across compared algorithms, clean/attack pairs, and local epochs.

A technical smoke test precedes the matrix. Its metrics are not scientific statistics.

## 7. Primary metrics and evaluation time

Primary metrics are computed on the locked test set at the pre-fixed final round: macro-F1; balanced accuracy; weighted/global accuracy; per-class recall; worst-class recall; 10th percentile of the class-recall distribution.

Attack primary contrasts: source-class recall harm; attacked macro-F1; clean utility penalty; attack recovery; operational visibility (harm minus accuracy drop).

Per-image predictions were archived in the laboratory tree so that paired and class-level tests remain possible. The public dump keeps confusion matrices and per-class metrics; the per-image lists stay laboratory-only.

## 8. Statistical plan

For each configuration, report mean, standard deviation, and a 95% bootstrap interval across independent seed runs. Algorithms and clean/attack conditions are compared as pairs:

- paired permutation test as the primary test;
- Wilcoxon signed-rank as sensitivity;
- paired difference and an interval for the effect;
- Holm correction inside a pre-specified family, not across families.

Class-level analysis uses entropy and monopoly/concentration. The planned mixed model is:

`class_recall ~ entropy + monopoly + algorithm + attack + monopoly:attack + algorithm:attack + (1 | seed) + (1 | class)`

The confirmatory coding of `attack` is documented in `docs/experiment_protocol/locked_inference_plan.md`.

## 9. Execution order

**Phase 0 — dataset audit.** Class normalisation without copying files; `leaf_id` coverage; exact-duplicate and perceptual reports; PV-19-capped / PV-19-full manifests; gate and reproducibility.

**Phase 1 — evaluation infrastructure.** Separate train partitions, development validation, and final test; post-run evaluator for saved global checkpoints; per-image predictions and run manifests (laboratory); real `FL_LOCAL_EPOCHS`, seed, and FedProx parameters in the launcher; protocol unit tests.

**Phase 2 — local smoke.** One small configuration (fewer rounds) only to check that manifests, training, aggregation, and the evaluator share the same class mapping.

**Phase 3 — PV-19-capped utility matrix.** Clean FedAvg/FedProx first. Attack/defense does not start until clean results, failures, and cost are audited.

**Phase 4 — attack/defense matrix.** At least three source/target pairs chosen in advance from class-concentration profiles, not from final-test scores.

**Phase 5 — PV-19-full confirmation.** Only the key configurations locked after the capped development protocol.

## 10. Deviations

**24 August 2026, before any new FL run.** The initial PV-27 intention was replaced by the primary PV-19 protocol. Official local leaf metadata are incomplete for all 27 selected classes. The exploratory PV-27-capped manifest had 1,200 images without a leaf ID; perceptual audit then found unresolved similar pairs that cross the split boundary, including visually confirmed shots of the same physical leaf. The perceptual-hash threshold also misses most known sibling images, so it was not used for automatic merging.

The primary set therefore keeps only classes with at least 99% class-aware leaf coverage and drops remaining images without a leaf ID. The resulting `pv19-capped-62b5b2119fb2` has 5,526 images, 993 groups, zero unknown leaf IDs, and passes the dataset gate. The change is a methodological correction made before any new model result was seen, not a post-hoc accuracy choice.

**31 August 2026, after the FL slices closed, before p-values.** The RQ1–RQ3 inference plan was locked in `docs/experiment_protocol/locked_inference_plan.md` and run on already-archived locked-test artefacts (`docs/manuscript/locked_inference.py`). No new training. In the mixed model, the protocol factor `attack` is coded as `targeted` at class-within-job (1 only if that class is the locked label-flip source). A job-level attack flag would mix targeted harm with collateral on the other 18 classes and would not answer RQ2. The analysis is confirmatory for families F1–F3 and the Gaussian LMM; binomial GEE is sensitivity (five seed clusters).

Later dated amendments (inferential unit, dose \(qM_c\), LMM audit, own-baseline harm, visibility-gap arithmetic, executable specification) are in the same plan file.

Each further deviation records the date, reason, affected runs, and whether the analysis is confirmatory or exploratory.
