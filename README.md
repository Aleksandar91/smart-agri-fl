# PV-19 class-concentrated federated learning — protocol artefact

Public archive of the locked evaluation protocol that starts on **24 August 2026**: a 19-class PlantVillage subset (PV-19-capped) and the PV-19-full follow-on slice.

Current tag: `pv19-protocol-v5`.

## Contents

| Path | Role |
|------|------|
| `fl-client/`, `fl-server/` | Flower 1.32.1 client and server used for the scored jobs |
| `infra/run_pv19_*.sh`, `run_fl_sequential.sh` | Matrix launchers (CPU Docker, five clients) |
| `infra/attack-configs/` | Locked apple / cherry / potato flips and Apple model-update |
| `docs/research_protocol_20260824.md` | Dataset and slice lock |
| `docs/experiment_protocol/` | Plan, attacker map, manifests, partitions, locked-test scores |
| `docs/manuscript/locked_inference.py` | Confirmatory tests and secondary analyses |
| `docs/experiment_protocol/attack_flip_counts.json` | Observed round-1 flipped-label counts |
| `docs/manuscript/figures/` | Figures 1–5 |

PlantVillage images are not in this repository. Point `PV19_RAW_COLOR` at a local copy of `raw/color` from [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset). Manifests list relative paths and SHA-256.

Scored jobs ran with differential privacy off and TLS off. The server still imports those optional helpers.

## Reproduce the published numbers

The analysis environment used for the published JSON is **Python 3.7** with `numpy`, `pandas`, `scipy`, and **statsmodels 0.12.2** (`weights=` is a GEE constructor argument on that build). Later statsmodels versions may reject `var_weights` on GEE and leave the binomial GEE unweighted.

```bash
python docs/manuscript/locked_inference.py
python docs/manuscript/plot_figures.py
```

`locked_inference.py` reads flip counts from `attack_flip_counts.json` when `fl_history.json` is absent.

Training the 675 jobs is a separate CPU Docker campaign. Launchers expect 128 px, batch 16, ten rounds, `FL_FREEZE_BACKBONE=1`, and five clients. Settings are in `docs/research_protocol_20260824.md` and `infra/run_pv19_*.sh`.

## Licence

Code: MIT. PlantVillage photographs remain under their upstream terms. This repository contains derived manifests and scores.
