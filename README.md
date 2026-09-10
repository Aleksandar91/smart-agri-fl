# PV-19 class-concentrated federated learning — protocol artefact

Public, versioned archive for the locked evaluation protocol on a 19-class
PlantVillage subset (PV-19-capped and the PV-19-full follow-on slice).

This repository is **not** the private laboratory monorepo. It does not
contain edge/Pi clients, TLS certificates, the web app, or differential-privacy
slices. Those remain internal.

Public GitHub repository: <https://github.com/Aleksandar91/smart-agri-fl>

## What is in here

| Path | Role |
|------|------|
| `fl-client/`, `fl-server/` | Scored Flower 1.32.1 client and server |
| `infra/run_pv19_*.sh`, `run_fl_sequential.sh` | Matrix launchers (CPU Docker) |
| `infra/attack-configs/` | Label-flip and model-update JSON |
| `docs/research_protocol_20260824.md` | Dataset and slice lock |
| `docs/experiment_protocol/` | Plan, attacker map, manifests, partitions, locked-test scores |
| `docs/manuscript/locked_inference.py` | Confirmatory tests and secondary analyses |
| `docs/manuscript/figures/` | Figures 1–5 |

PlantVillage **images are not redistributed**. Point `DATASET_DIR` at a local
copy of `raw/color` from [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)
(or the Hughes / Mohanty colour tree used in the lock). Manifests list relative
paths and SHA-256.

## What was stripped

- Checkpoints (`*.npz`) — SHA-256 of each scored `global_latest.npz` is in
  `test_evaluation.json` as `checkpoint_sha256`.
- Per-image prediction lists (`predictions` in the original evaluator output).
  Confusion matrices and per-class metrics are kept; they are what the paper
  tables and `locked_inference.py` use.
- Development-validation JSON and per-round `fl_history.json` (available from
  the laboratory archive on request).
- Absolute Windows paths from the laboratory machine (`<repository-root>`).

## Reproduce the published numbers

Python 3.7+ with `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`.

```bash
python docs/manuscript/locked_inference.py
python docs/manuscript/plot_figures.py
```

Training the 675 jobs is a separate, long CPU Docker campaign. Launchers expect
the same environment variables as in the paper (128 px, batch 16, ten rounds,
`FL_FREEZE_BACKBONE=1`). See `docs/manuscript/class_concentrated_fl_pv19_en.md`
in the laboratory tree, or the methods section of the submitted manuscript.

## Licence

Code: MIT (this repository). PlantVillage photographs remain under their
upstream terms; this dump only contains derived manifests and scores.
