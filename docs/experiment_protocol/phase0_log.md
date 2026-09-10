# Phase 0–1 log — dataset audit and evaluation infrastructure

## 24 August 2026

### 0.1 Local PlantVillage source

Confirmed local original PlantVillage tree. Inventory:

- `raw/color`: 54,305 RGB files, 38 classes;
- `raw/grayscale` / `raw/segmented`: same class count;
- `leaf-map.json`: 40,328 keys;
- `leaf_grouping/filtered_leafmaps`: class-level CSV maps of physical leaves.

This protocol uses only `raw/color`. PlantVillage was not re-downloaded.

### 0.2 Audit of the previous pipeline

The previous (pre-protocol) pipeline had two methodological weaknesses:

1. `partition_dataset.py` partitioned all images directly to clients, without a locked global test set;
2. `fl_task.py` then made a per-image local train/validation split, without using `leaf_id` groups.

Different shots of the same physical leaf could therefore land in train and validation. Those earlier results remain engineering evidence and are **not** confirmatory results of this study.

### 0.3 Leaf-metadata coverage

`leaf-map.json` is not complete for all 38 classes. Filename-only lookup is unsafe because the same original filename can exist in more than one class. Apple black rot uses the documented synonym `Apple_Frogeye Spot`. A class-aware lookup was implemented (accept a leaf ID only from the matching class).

### 0.4 Protocol manifest generator

Added `fl-client/app/prepare_pv_protocol.py`: does not copy or alter images; normalises class names; SHA-256 of each source image; links class-aware leaf siblings and exact duplicates; capped variant at whole-group level; one global 70/15/15 split before FL partitioning; checks group and SHA-256 intersections; writes `samples` and per-image `records`; keeps the dataset gate closed until the perceptual audit finishes.

Unit tests: `fl-client/tests/test_prepare_pv_protocol.py`.

### 0.5 Dry-run and source aliases

The first real run was stopped without an output manifest because the original tree uses different names for ten tomato directories than an earlier Kaggle mirror. An explicit source-alias map and a fail-fast directory audit were added. Missing classes are not silently skipped.

### 0.6 Exploratory PV-27-capped (not confirmatory)

Artefacts were written under `pv27-capped-v1` in the laboratory tree. They are **not** in this public dump.

Result used only to reject PV-27 as confirmatory: 7,926 images; 1,200 without a known leaf ID; group and SHA-256 boundary checks passed; an independent rerun matched the manifest hash.

### 0.7 Perceptual audit of exploratory PV-27

Added `fl-client/app/audit_pv_perceptual.py` (64-bit pHash and dHash; BK-tree retrieval; never auto-merges). On PV-27-capped: 20 unresolved pairs because at least one side lacks a leaf ID; 11 of those 20 cross a split boundary. Visual checks of top-ranked candidates (e.g. `Grape___healthy` 9127/9129) show the same physical leaf in slightly different pose, assigned to different splits.

Among 21,352 known sibling pairs the median pHash distance is 26; only 207 pairs enter a radius-8 candidate set. Automatic merge by threshold was rejected. The exploratory PV-27 gate stays closed.

### 0.8 Strict primary PV-19-capped

Before any new FL run the protocol was narrowed: a class needs ≥99% class-aware leaf-ID coverage; after the class filter every remaining image must have a leaf ID.

Artefacts: `docs/experiment_protocol/datasets/pv19-capped-primary-v1`

- `dataset_id`: `pv19-capped-62b5b2119fb2`
- manifest SHA-256: `62b5b2119fb2e12b004e59c5fdfdef29765e2e4bb9c1b6cf79aeb3155ba8f872`
- 19 classes; 5,526 images; 993 physical leaf groups; 0 images without leaf ID
- 7 exact-duplicate sets, absorbed into existing groups
- perceptual candidates between different groups: 72, all resolved by distinct official leaf IDs; unresolved: 0
- dataset gate: **pass**

PV-27 → PV-19 was decided before any new model result was seen. The reason is leakage control, not accuracy optimisation.

### 0.9 PV-19-full confirmatory dataset

Artefacts: `docs/experiment_protocol/datasets/pv19-full-confirmatory-v1`

- `dataset_id`: `pv19-full-774007483a1d`
- manifest SHA-256: `774007483a1d71af339ae430d2e41dfd90fb966363c18c238dc0dc97004626d2`
- 20,597 images; 4,064 physical leaf groups
- train 14,372; development validation 3,115; final test 3,110
- images without leaf ID: 0; unresolved perceptual candidates: 0
- dataset gate: **pass**

The full utility/attack matrix is not run on this set. Its role is a pre-limited confirmation of key PV-19-capped findings.

### 0.10 Group-safe FL partitioning of the train pool

`partition_dataset.py --source-manifest`: accepts only a manifest whose split is `train`; copies dataset ID and manifest SHA-256 into each client file; assigns whole leaf groups; checks no group crosses a client boundary and that every source group is assigned once; writes class-level monopoly, normalised entropy, effective-client and client sample-share measures.

The legacy `--data-root` mode exists only to reproduce older laboratory runs. It is **not** used in this confirmatory study.

First check: `partitions/pv19-capped/a05/seed-101` — dataset `pv19-capped-62b5b2119fb2`; five clients; Dirichlet α = 0.5; seed 101; 3,797 train images; 685 leaf groups, all assigned once. Client sizes 556, 551, 757, 729, 1,204. Mean class monopoly 0.611; mean normalised class entropy 0.629. Alpha alone does not describe realised class-level heterogeneity.

## Phase 1 — evaluation infrastructure and technical smoke

### 1.1 Training vs development-validation

`FL_EVAL_MANIFEST`: the client trains on all images in its group-safe train partition; round evaluation uses only global `validation.json`; `test.json` is rejected by the FL round evaluator. Dataset ID, class order, and path intersection of train/validation are checked. `FL_EVAL_CLIENTS=1`: one evaluator per round.

### 1.2 Independent post-run evaluator

`evaluate_global.py` accepts only a locked validation or test manifest; loads a specified Flower `.npz`; deterministic evaluation transform; accuracy, balanced accuracy, macro-F1, weighted-F1, worst-class recall, 10th-percentile recall, per-class metrics, confusion matrix; archives per-image target/prediction/confidence in the laboratory tree; records checkpoint SHA-256, dataset-manifest SHA-256, and environment versions.

A technical check on the initial 19-class checkpoint processed all 879 validation images. Those metrics are not a research result.

### 1.3 Flower evaluation configuration

A 1-round run trained all five clients but the server logged `configure_evaluate: no clients selected, skipping evaluation` because Flower skips evaluation when `fraction_evaluate=0` even if `min_evaluate_clients=1`. Fixed to `fraction_evaluate = eval_clients / min_clients`. The launcher now fails a run if a successful locked validation evaluation is missing for any round.

### 1.4 Protocol smoke (not scientific)

One-round FedAvg on α=0.5 seed 101: 3,797 train images; 879 global validation images; fit/eval failures 0. Round-1 development-validation accuracy 53.47% (macro-F1 48.92%). Independent post-run evaluator matched accuracy `0.534698521046644`. Not entered into multi-seed statistics; final test not scored.

### 1.5 FedProx smoke and clean-matrix preparation

Flower `FedProx` with required `proximal_mu`. The client adds `0.5 μ ||w − w_global||²` only on trainable parameters. Locked value for the primary matrix: `μ = 0.01`. One-round FedProx smoke on the same partition: development-validation accuracy 54.15%. Not a scientific result.

Group-safe partitions for the whole clean matrix:

`docs/experiment_protocol/partitions/pv19-capped/{a01,a05,iid}/seed-{101,211,307,401,503}`

Mean class monopoly (five seeds), approximate:

- α=0.1: about 0.75–0.81;
- α=0.5: about 0.50–0.61;
- IID: about 0.22, with normalised entropy ≈ 0.998.

Pre-registered clean jobs: 60 (3 distributions × 2 algorithms × 2 local epochs × 5 seeds, 10 rounds each).
