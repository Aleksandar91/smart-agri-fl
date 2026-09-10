# PV-19 class-concentrated federated learning — protocol artefact

Public, versioned, **English-language** archive for the locked evaluation
protocol that starts on **24 August 2026**: a 19-class PlantVillage subset
(PV-19-capped) and the PV-19-full follow-on slice.

This repository is **not** the private laboratory monorepo and **not** the
earlier PlantVillage-v2 / Raspberry Pi / differential-privacy study. It does
not contain edge/Pi clients, TLS certificates, the web app, DP analysis
scripts, tomato label-flip configs from that campaign, or Serbian laboratory
notes.

Public GitHub repository: <https://github.com/Aleksandar91/smart-agri-fl>

## What is in here

| Path | Role |
|------|------|
| `fl-client/`, `fl-server/` | Scored Flower 1.32.1 client and server (PV-19 import graph) |
| `infra/run_pv19_*.sh`, `run_fl_sequential.sh` | Matrix launchers (CPU Docker, five clients) |
| `infra/attack-configs/` | Locked apple / cherry / potato flips and Apple model-update |
| `docs/research_protocol_20260824.md` | Dataset and slice lock |
| `docs/experiment_protocol/` | Plan, attacker map, manifests, partitions, locked-test scores |
| `docs/manuscript/class_concentrated_fl_pv19_en.md` | English manuscript draft |
| `docs/manuscript/locked_inference.py` | Confirmatory tests and secondary analyses |
| `docs/manuscript/figures/` | Figures 1–5 |

PlantVillage **images are not redistributed**. Point `PV19_RAW_COLOR` at a
local copy of `raw/color` from [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)
(or the Hughes / Mohanty colour tree used in the lock). Manifests list relative
paths and SHA-256.

## What was stripped

- Checkpoints (`*.npz`) — SHA-256 of each scored `global_latest.npz` is in
  `test_evaluation.json` as `checkpoint_sha256`.
- Per-image prediction lists (`predictions` in the original evaluator output).
  Confusion matrices and per-class metrics are kept; they are what the paper
  tables and `locked_inference.py` use.
- Development-validation JSON and per-round `fl_history.json` (laboratory).
- USB-camera, FastAPI ping, personalization, synthetic-data, and DP-report
  modules from the earlier campaign.
- Absolute Windows paths (`<repository-root>`).

`fl_server.py` still imports optional DP / TLS helpers because the scored
process does. PV-19 jobs ran with differential privacy off and TLS off.

## Reproduce the published numbers

Python 3.7+ with `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`.

```bash
python docs/manuscript/locked_inference.py
python docs/manuscript/plot_figures.py
```

Training the 675 jobs is a separate, long CPU Docker campaign. Launchers expect
the same environment variables as in the paper (128 px, batch 16, ten rounds,
`FL_FREEZE_BACKBONE=1`, five clients). See `docs/manuscript/class_concentrated_fl_pv19_en.md`.

## Licence

Code: MIT (this repository). PlantVillage photographs remain under their
upstream terms; this dump only contains derived manifests and scores.
