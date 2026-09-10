# Phase 4 log — attack/defense matrix

## 25 August 2026

Phase 3 was complete: 60/60 clean runs, 0 failures, locked test scored. Attack pairs remain those locked on 24 August from partitions, not from test scores.

### Stage 4.1 — primary label-flip slice

Pre-specified subset only:

- three locked pairs (apple, cherry, potato);
- flip fraction `1.0`;
- FedAvg and FedProx, to pair with clean E=1 runs;
- E=1, 10 rounds, 5 clients;
- α=0.1, 0.5, IID and seeds 101, 211, 307, 401, 503;
- attacker = client with the most source-class train images on that partition.

90 jobs. The final test is not scored until 4.1 finishes.

Not in 4.1: flip 0.25 and 0.50; model-update `s = −0.5` and `s = −1`; median, trimmed mean, Krum/MultiKrum; E=5.

Canonical gate: `pv19-capped-a05-s101-fedavg-e1-flip1-apple` (attacker `client-4`). Passed 25 August 2026 12:41–12:48 UTC (~6.4 min): 10/10 rounds; label-flip only on `client-4` in all 10 rounds (`label_flipped_samples=124` per round); locked validation 879 images, accuracy 0.807, macro-F1 0.804; `test_evaluated=false`.

## 26 August 2026

The supervising process lost contact with the launcher at 25 August 21:49 UTC, after Docker had started `pv19-capped-iid-s503-fedavg-e1-flip1-apple`. The launcher itself was not killed: Docker stayed on that job ~8 h (same stall pattern as phase 3), then finished 10/10 rounds and sealed validation at 26 August 05:56 UTC. `test_evaluated=false`.

By then 85/90 jobs were done. The launcher continued the remaining five. A second launcher was not started, to avoid colliding containers.

All 90 jobs have 10/10 rounds, `global_latest.npz`, and `validation_evaluation.json`. Last: `pv19-capped-iid-s503-fedprox-e1-flip1-potato` at 06:29 UTC.

Audit: `phase4_audit.json` — 90/90, 0 fit/eval failures; label-flip only on the locked attacker; typical duration 6.5 min (5.5–7.5); one stall `iid-s503-fedavg-e1-flip1-apple` ~8.1 h (supervision/Docker, not extra training); training wall-clock 25 August 12:41 UTC to 26 August 06:29 UTC.

`PV19_ATTACK_MODE=eval_test` finished on 26 August. All 90 have `test_evaluation.json` (850 images, `split=test`). Summary: `phase4_test_summary.json`. Those numbers do not change pairs, attackers, or 4.2a aggregators.

### Stage 4.2a — robust aggregators at flip 1.0

Same partitions, pairs, attackers, E=1 and 10 rounds as 4.1. Treatment is the server aggregator: median, trimmed mean, Krum, MultiKrum. 180 jobs.

Canonical gate: `pv19-capped-a05-s101-median-e1-flip1-apple` (attacker `client-4`). Passed 26 August 07:31–07:38 UTC.

Around 14:11 UTC a laptop restart interrupted Docker on `pv19-capped-a05-s101-median-e1-flip1-cherry` (3/10 rounds, no validation). 61/180 were sealed; the incomplete artefact was deleted. The launcher resumed from that job and skips complete ones.

Not in 4.2a: flip 0.25/0.50, model-update, E=5, clean robust runs.

## 27 August 2026

All 180 jobs have 10/10 rounds, checkpoint, and validation. Last: `pv19-capped-iid-s503-multikrum-e1-flip1-potato` at 14:07 UTC. Audit: `phase4_defense_audit.json` — 0 fit/eval failures; typical ~7 min; two stalls (`iid-s101-krum-apple` ~8.4 h, `iid-s401-krum-apple` ~80 min); wall-clock 26 August 07:31 UTC to 27 August 14:07 UTC.

`PV19_DEFENSE_MODE=eval_test` finished 27 August 14:35 UTC. All 180 have `test_evaluation.json` (850 images). Summary: `phase4_defense_test_summary.json`. Those numbers do not change 4.2b.

### Stage 4.2b — flip 0.25 and 0.50

Same partitions, pairs, attackers, FedAvg/FedProx, E=1 as 4.1. 180 jobs.

At 19:02 UTC the Docker daemon stopped during `pv19-capped-a01-s401-fedprox-e1-flip050-apple` (launcher exit 127, no artefact). 43/180 were sealed. The daemon was restarted; the launcher skips complete jobs.

The launcher was paused at 55/180 that evening.

## 28 August 2026

4.2b resumed. All 180 jobs have 10/10 rounds. Last: `pv19-capped-iid-s503-fedprox-e1-flip050-potato` at 20:58 UTC. Audit: `phase4_flip_sensitivity_audit.json` — 0 fit/eval failures; typical ~6.4 min (5.3–7.8); training stalls: 0 (27–28 August interruptions recovered by skip-complete resume); training wall-clock 27 August 14:37 UTC to 28 August 20:58 UTC.

`PV19_FLIP_SENS_MODE=eval_test` finished 28 August 21:27 UTC. All 180 have `test_evaluation.json`. Summary: `phase4_flip_sensitivity_test_summary.json`. Those numbers do not change 4.2c (model-update and clean robust were already locked before this test was inspected).

### Stage 4.2c — model-update, then clean robust

First 60 jobs: scale `s = −0.5` and `s = −1` × FedAvg/FedProx × 3 α × 5 seeds. Attacker = locked monopoly client for `Apple___healthy`. Then 60 clean jobs of the robust aggregators (no attack).

## 29 August 2026

All 60 model-update jobs finished. Last: `pv19-capped-iid-s503-fedprox-e1-muminus1` at 03:52 UTC. Audit: `phase4_model_update_audit.json` — 0 failures; model-update only on the locked Apple attacker, scale correct in 10/10 rounds; typical ~6.3 min; wall-clock 28 August 21:32 UTC to 29 August 03:53 UTC.

`PV19_MU_MODE=eval_test` finished 29 August 04:03 UTC. Summary: `phase4_model_update_test_summary.json`.

Clean robust: 60 jobs, median / trimmed mean / Krum / MultiKrum × 3 α × 5 seeds, E=1, no attack. Last: `pv19-capped-iid-s503-multikrum-e1-clean` at 10:28 UTC. Audit: `phase4_clean_robust_audit.json` — 0 failures; unexpected attacks: 0; typical ~6.3 min; wall-clock 29 August 04:08–10:28 UTC.

`PV19_CLEAN_ROBUST_MODE=eval_test` finished ~10:40 UTC. Summary: `phase4_clean_robust_test_summary.json`. Those numbers do not change phase 5 (PV-19-full was already locked).

Phase 5 started: 45 jobs, E=1, 10 rounds. First: `pv19-full-a01-s101-fedavg-e1`.

## 30 August 2026

All 45 phase-5 jobs finished. Last: `pv19-full-iid-s503-median-e1-flip1-apple` at 03:18 UTC. Audit: `phase5_full_audit.json` — 0 failures; flip only on the full Apple attacker; clean jobs without attack; typical ~22 min (18–25); wall-clock 29 August 10:42 UTC to 30 August 03:18 UTC.

`PV19_FULL_MODE=eval_test` finished 30 August 03:35 UTC. All 45 have `test_evaluation.json` (3,110 images). Summary: `phase5_full_test_summary.json`.

E=5 under attack was not run.
