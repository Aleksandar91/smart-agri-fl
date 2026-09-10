# Manuscript figures 1–5

Regenerate from locked phase 3–4.2c summaries (do not re-type table cells):

```
python docs/manuscript/plot_figures.py
```

Each figure is written as a vector PDF (fonttype 42) and a 600 dpi PNG.

| File | Role |
|------|------|
| `fig1_testbed` | Locked protocol: split, five silos, honest aggregator, test scored after each slice |
| `fig2_class_client_heatmap` | Class shares and \(M_c\) for seed 101 at \(\alpha=0.1\), \(0.5\), IID |
| `fig3_monopoly_harm` | Train monopoly vs paired source-recall harm within each α (phase 4.1, three panels, \(n=30\) each) |
| `fig4_visibility_gap` | Accuracy drop vs source harm at flip 0.25 / 0.50 / 1.0 (FedAvg) |
| `fig5_aggregator_pareto` | Clean macro-F1 vs attack-induced source harm (own clean baseline) |

Captions live in the EN/SR markdown files next to the image includes.
