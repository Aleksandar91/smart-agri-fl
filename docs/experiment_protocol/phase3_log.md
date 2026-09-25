# Phase 3 log — clean utility matrix

## 24 August 2026

### Gate before launch

Phases 0–2 were complete. Smoke runs are not scientific. Before the first 10-round run, the following were locked:

- dataset `pv19-capped-62b5b2119fb2`;
- 15 group-safe partitions (α=0.1, 0.5, IID × 5 seeds);
- 60 clean jobs: FedAvg/FedProx × E=1/E=5 × 10 rounds;
- FedProx `μ = 0.01`;
- development-validation during rounds, one evaluator per round;
- the final test is **not** scored until all 60 runs finish (`PV19_MATRIX_MODE=eval_test`);
- three phase-4 label-flip pairs, from monopoly measures, without inspecting the test.

### Canonical job (technical audit, not algorithm choice)

`docs/experiment_protocol/runs/pv19-capped/pv19-capped-a01-s101-fedavg-e1`

- duration: 17:23:35–17:30:56 UTC (7 min 21 s);
- 10/10 rounds with locked validation evaluation;
- fit and evaluation failures: 0;
- final test not scored;
- development-validation accuracy in round 10: 56.20% (macro-F1 49.52% on the independent evaluator).

These numbers check the pipeline and cost only. They are not used to choose α, E, aggregator, or attack. Remaining-matrix estimate: about 22 hours on the same CPU Docker path (29×E=1 + 30×E=5). The other 59 jobs started immediately after this check. The script skips already-finished jobs.

## 25 August 2026

By 05:34 UTC, 41/60 jobs were done: all α=0.1, all α=0.5, and `iid-s101-fedavg-e1`. The final test was not scored on any run.

The parent `run_pv19_utility_matrix.sh` process stopped while Docker continued `pv19-capped-iid-s101-fedavg-e5`. That job was not killed or restarted. After its `docker wait`, only the validation evaluator and `run_manifest` ran, then the remaining 18 jobs with the same script.

The current E=5 job finished in Docker (10/10 rounds, 0 failures). The launcher resumed at 06:52 UTC and skipped 42 finished jobs.

### Clean-matrix completion and audit gate

All 60 pre-registered jobs finished. Audit artefact (laboratory): `phase3_audit.json`.

Checks:

- 60/60 have 10/10 rounds, `global_latest.npz`, and `validation_evaluation.json`;
- total fit failures: 0;
- total evaluation failures: 0;
- `test_evaluation.json` files at that moment: 0;
- wall-clock: 2026-08-24 17:23 UTC to 2026-08-25 12:23 UTC (~19 h).

Typical duration (excluding two Docker stalls): E=1 mean 6.5 min (5.5–10); E=5 mean 31.4 min, or about 23–29 min without two stalls.

Two wall-clock outliers were not extra training: `a01-s101-fedprox-e5` ~100 min (overnight stall); `iid-s101-fedavg-e5` ~73 min (launcher stop + pauses in rounds 5 and 8).

Development-validation, as a pipeline sanity check only, follows the expected heterogeneity order (IID > α=0.5 > α=0.1). Primary metrics remain on the locked test, which is scored only because the matrix is complete.

### Final test — batch evaluation

`PV19_MATRIX_MODE=eval_test` finished 25 August 12:33 UTC. All 60 checkpoints were scored on the locked test (850 images). Attack pairs were not changed.

Primary test metrics (mean ± sd, 5 seeds):

- α=0.1 FedAvg E=1: accuracy 0.636 ± 0.061; macro-F1 0.591 ± 0.076
- α=0.1 FedAvg E=5: 0.702 ± 0.050; 0.665 ± 0.068
- α=0.1 FedProx E=1: 0.678 ± 0.045; 0.649 ± 0.055
- α=0.1 FedProx E=5: 0.753 ± 0.042; 0.737 ± 0.049
- α=0.5 FedAvg E=1: 0.809 ± 0.032; 0.805 ± 0.036
- α=0.5 FedAvg E=5: 0.842 ± 0.028; 0.839 ± 0.030
- α=0.5 FedProx E=1: 0.821 ± 0.021; 0.820 ± 0.023
- α=0.5 FedProx E=5: 0.852 ± 0.009; 0.851 ± 0.010
- IID FedAvg E=1: 0.861 ± 0.005; 0.861 ± 0.006
- IID FedAvg E=5: 0.879 ± 0.005; 0.878 ± 0.005
- IID FedProx E=1: 0.857 ± 0.005; 0.856 ± 0.005
- IID FedProx E=5: 0.874 ± 0.002; 0.874 ± 0.003

Detail: `docs/experiment_protocol/runs/pv19-capped/phase3_test_summary.json`.

Phase 4 starts on the pairs locked from partitions, not on the basis of these numbers.
