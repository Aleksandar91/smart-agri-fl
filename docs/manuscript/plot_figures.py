"""Draw the five locked manuscript figures from archived test summaries.

Outputs PDF (vector, fonttype 42) and 600 dpi PNG under docs/manuscript/figures/.
Numbers are read from phase 3–4.2c JSON; they are not re-typed from the tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "experiment_protocol"
MS = Path(__file__).resolve().parent
FIGDIR = MS / "figures"
PART = PROTOCOL / "partitions" / "pv19-capped"
RUNS = PROTOCOL / "runs"

SEED_HEATMAP = 101
SOURCE_CLASSES = {
    "apple": "Apple___healthy",
    "cherry": "Cherry___healthy",
    "potato": "Potato___healthy",
}
ALPHA_ORDER = ("a01", "a05", "iid")
ALPHA_LABEL = {"a01": r"$\alpha=0.1$", "a05": r"$\alpha=0.5$", "iid": "IID"}
PAIR_ORDER = ("apple", "cherry", "potato")
PAIR_LABEL = {"apple": "Apple", "cherry": "Cherry", "potato": "Potato"}
AGG_ORDER = ("fedavg", "trimmed_mean", "median", "multikrum", "krum")
AGG_LABEL = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed mean",
    "median": "Median",
    "multikrum": "MultiKrum",
    "krum": "Krum",
}
CLASS_SHORT = {
    "Apple___Apple_scab": "Apple scab",
    "Apple___Black_rot": "Apple black rot",
    "Apple___Cedar_apple_rust": "Apple cedar rust",
    "Apple___healthy": "Apple healthy",
    "Blueberry___healthy": "Blueberry healthy",
    "Cherry___Powdery_mildew": "Cherry mildew",
    "Cherry___healthy": "Cherry healthy",
    "Grape___Black_rot": "Grape black rot",
    "Grape___Esca": "Grape esca",
    "Grape___Leaf_blight": "Grape blight",
    "Pepper___Bacterial_spot": "Pepper bacterial",
    "Pepper___healthy": "Pepper healthy",
    "Potato___Early_blight": "Potato early blight",
    "Potato___Late_blight": "Potato late blight",
    "Potato___healthy": "Potato healthy",
    "Tomato___Bacterial_spot": "Tomato bacterial",
    "Tomato___Early_blight": "Tomato early blight",
    "Tomato___Leaf_Mold": "Tomato leaf mold",
    "Tomato___Spider_mites": "Tomato mites",
}

# Okabe–Ito (colorblind-safe)
C = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
    "light": "#F4F4F4",
}
ALPHA_COLOR = {"a01": C["vermillion"], "a05": C["orange"], "iid": C["blue"]}
PAIR_MARKER = {"apple": "o", "cherry": "s", "potato": "^"}
PAIR_COLOR = {"apple": C["green"], "cherry": C["purple"], "potato": C["blue"]}
AGG_MARKER = {
    "fedavg": "D",
    "trimmed_mean": "o",
    "median": "s",
    "multikrum": "P",
    "krum": "X",
}

SHARE_CMAP = LinearSegmentedColormap.from_list(
    "share_oi",
    ["#FFFFFF", "#F0E442", "#E69F00", "#D55E00", "#7A2E0E"],
    N=256,
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "mathtext.fontset": "dejavusans",
        }
    )


def save(fig: mpl.figure.Figure, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGDIR / f"{stem}.pdf"
    png = FIGDIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"wrote {pdf.name} and {png.name}")


def panel_label(ax, text: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def partition_summary(alpha: str, seed: int) -> dict:
    return load_json(PART / alpha / f"seed-{seed}" / "partitions_summary.json")


def class_metrics_of(summary: dict) -> dict:
    if "class_metrics" in summary:
        return summary["class_metrics"]
    return summary["heterogeneity"]["class_metrics"]


def attacker_table() -> dict:
    raw = load_json(PROTOCOL / "attack_attacker_clients.json")
    table: dict[tuple[str, str, int], int] = {}
    for pair in raw["pairs"]:
        key = {
            "Apple___healthy": "apple",
            "Cherry___healthy": "cherry",
            "Potato___healthy": "potato",
        }[pair["source"]]
        for alpha, seeds in pair["attackers"].items():
            for seed, client in seeds.items():
                table[(alpha, key, int(seed))] = int(client)
    return table


def fig1_testbed() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)

    def box(x, y, w, h, text, facecolor="#FFFFFF", edge="#333333", lw=0.9, ls="-", fs=7.5, weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=facecolor,
            edgecolor=edge,
            linewidth=lw,
            linestyle=ls,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color="#111111",
            fontweight=weight,
            zorder=3,
            wrap=True,
        )
        return patch

    def arrow(x1, y1, x2, y2, color="#333333"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.8,
                color=color,
                zorder=1,
            )
        )

    box(3.15, 8.55, 5.7, 1.15, "PV-19-capped  ·  5,526 images  ·  19 classes  ·  993 leaf groups\nleaf-safe, hash-locked  ·  dataset_id  pv19-capped-62b5b2119fb2", "#EEF6FB", C["blue"], fs=7.5, weight="bold")

    arrow(6.0, 8.55, 6.0, 7.85)
    ax.text(6.15, 8.15, "group-safe split  (no leaf group or SHA-256 crosses splits)", fontsize=6.5, color=C["gray"], va="center")

    box(0.35, 6.35, 3.5, 1.35, "Train pool\n3,797 images  ·  70%\nclient partitions drawn here only", "#E8F6F3", C["green"], fs=7.5)
    box(4.25, 6.35, 3.5, 1.35, "Development validation\n879 images  ·  15%\nused during FL rounds", C["light"], "#555555", fs=7.5)
    box(8.15, 6.35, 3.5, 1.35, "Final test  ·  SEALED\n850 images  ·  15%\nevaluated once per finished slice", "#FDEBD0", C["vermillion"], fs=7.5, weight="bold")

    arrow(2.1, 6.35, 2.1, 5.55)
    ax.text(2.25, 5.85, r"Dirichlet  $\alpha$ = 0.1 or 0.5,  or IID  ·  five partition seeds", fontsize=6.5, color=C["gray"], va="center")

    client_w, client_h = 1.9, 1.28
    gap = 0.22
    x0 = 0.45
    y_c = 3.95
    xs = []
    for i in range(5):
        x = x0 + i * (client_w + gap)
        xs.append(x + client_w / 2)
        box(
            x,
            y_c,
            client_w,
            client_h,
            f"Client {i}\nlocal MobileNet head\nimages stay local",
            "#FFFFFF",
            "#444444",
            fs=6.5,
        )
    ax.add_patch(
        FancyBboxPatch(
            (0.32, 3.82),
            11.36,
            1.55,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor="none",
            edgecolor=C["vermillion"],
            linewidth=1.05,
            linestyle="--",
            zorder=1,
        )
    )
    ax.text(
        6.0,
        5.50,
        "five software silos  ·  images never leave the client process",
        ha="center",
        fontsize=6.5,
        color=C["gray"],
    )
    ax.text(
        6.0,
        3.58,
        "Attacker (exactly one, not a fixed index): client with the most source-class\ntraining images on that partition; ties broken by lowest index.",
        ha="center",
        va="center",
        fontsize=6.3,
        color=C["vermillion"],
        fontweight="bold",
    )

    for xc in xs:
        arrow(xc, 3.82, 6.0, 2.95)

    box(
        2.35,
        1.55,
        7.3,
        1.35,
        "Honest aggregator\nFedAvg  ·  FedProx  ·  coordinate median  ·  trimmed mean  ·  Krum  ·  MultiKrum\nparameter updates only  ·  validation/test labels are not flipped",
        "#EAF2F8",
        C["blue"],
        fs=7.5,
        weight="bold",
    )
    arrow(6.0, 1.55, 6.0, 0.95)
    box(3.35, 0.18, 5.3, 0.72, "Locked test metrics after each matrix slice  ·  not used for selection", "#FDEBD0", C["vermillion"], fs=7.2)

    ax.text(
        0.18,
        4.60,
        "Threat\nboundary",
        fontsize=6.2,
        color=C["vermillion"],
        fontweight="bold",
        va="center",
        ha="center",
        rotation=90,
    )
    save(fig, "fig1_testbed")


def fig2_heatmap() -> None:
    attackers = attacker_table()
    summaries = {alpha: partition_summary(alpha, SEED_HEATMAP) for alpha in ALPHA_ORDER}
    classes = summaries["a01"]["classes"]
    n_cls, n_cli = len(classes), 5

    fig = plt.figure(figsize=(7.2, 5.55))
    gs = fig.add_gridspec(
        1,
        7,
        width_ratios=[1.0, 0.16, 1.0, 0.16, 1.0, 0.16, 0.06],
        wspace=0.18,
        left=0.16,
        right=0.94,
        top=0.90,
        bottom=0.11,
    )
    heat_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 4])]
    mono_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 5])]
    cax = fig.add_subplot(gs[0, 6])

    im = None
    for col, alpha in enumerate(ALPHA_ORDER):
        metrics = class_metrics_of(summaries[alpha])
        shares = np.array([metrics[name]["shares"] for name in classes], dtype=float)
        monopoly = np.array([metrics[name]["monopoly"] for name in classes], dtype=float)
        ax = heat_axes[col]
        im = ax.imshow(shares, cmap=SHARE_CMAP, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(n_cli))
        ax.set_xticklabels([str(i) for i in range(n_cli)])
        ax.set_yticks(range(n_cls))
        short = [CLASS_SHORT[name] for name in classes]
        if col == 0:
            ax.set_yticklabels(short, fontsize=6.3)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Client")
        ax.set_title(ALPHA_LABEL[alpha], pad=6)
        ax.tick_params(length=0)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_linewidth(0.4)
        ax.spines["right"].set_linewidth(0.4)
        for spine in ax.spines.values():
            spine.set_color("#888888")

        for pair_key, class_name in SOURCE_CLASSES.items():
            row = classes.index(class_name)
            client = attackers[(alpha, pair_key, SEED_HEATMAP)]
            ax.add_patch(
                Rectangle(
                    (client - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=C["black"],
                    linewidth=1.15,
                    zorder=5,
                )
            )
            label = ax.get_yticklabels()[row] if col == 0 else None
            if label is not None:
                label.set_fontweight("bold")
                label.set_color(C["vermillion"])

        mx = mono_axes[col]
        mx.imshow(monopoly[:, None], cmap=SHARE_CMAP, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
        mx.set_xticks([])
        mx.set_yticks([])
        mx.set_xlabel(r"$M_c$", fontsize=7.5)
        mx.spines["top"].set_visible(True)
        mx.spines["right"].set_visible(True)
        for spine in mx.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#888888")
        for pair_key, class_name in SOURCE_CLASSES.items():
            row = classes.index(class_name)
            mx.text(
                0,
                row,
                f"{monopoly[row]:.2f}",
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if monopoly[row] > 0.55 else "black",
                fontweight="bold",
            )

        panel_label(ax, f"({chr(ord('a') + col)})", x=-0.04 if col else -0.08, y=1.05)

    fig.colorbar(im, cax=cax, label="Class share")
    cax.tick_params(labelsize=7)
    fig.text(0.55, 0.02, "Outlined cell = provisioned attacker for that source class (seed 101). Bold rows = locked flip sources.", ha="center", fontsize=6.5, color=C["gray"])
    save(fig, "fig2_class_client_heatmap")


def fig3_scatter() -> None:
    attack = load_json(RUNS / "pv19-capped-attack" / "phase4_test_summary.json")
    by_alpha = {alpha: {"x": [], "y": []} for alpha in ALPHA_ORDER}
    points = []
    for run in attack["runs"]:
        summary = partition_summary(run["alpha_label"], run["seed"])
        monopoly = class_metrics_of(summary)[run["source_class"]]["monopoly"]
        harm = run["source_recall_harm"]
        points.append((run["alpha_label"], run["pair"], run["algorithm"], monopoly, harm))
        by_alpha[run["alpha_label"]]["x"].append(monopoly)
        by_alpha[run["alpha_label"]]["y"].append(harm)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), sharey=True)
    for col, alpha in enumerate(ALPHA_ORDER):
        ax = axes[col]
        for a_label, pair, algorithm, monopoly, harm in points:
            if a_label != alpha:
                continue
            filled = algorithm == "fedavg"
            ax.scatter(
                monopoly,
                harm,
                s=26,
                marker=PAIR_MARKER[pair],
                facecolors=PAIR_COLOR[pair] if filled else "none",
                edgecolors=PAIR_COLOR[pair],
                linewidths=0.8,
                alpha=0.85,
                zorder=3,
            )
        xs = by_alpha[alpha]["x"]
        ys = by_alpha[alpha]["y"]
        rx = np.argsort(np.argsort(xs))
        ry = np.argsort(np.argsort(ys))
        rho = float(np.corrcoef(rx, ry)[0, 1])
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.08, 1.08)
        ax.axhline(0, color="#CCCCCC", linewidth=0.5, zorder=0)
        ax.set_title(ALPHA_LABEL[alpha])
        ax.set_xlabel(r"Source monopoly $M_c$")
        if col == 0:
            ax.set_ylabel("Source-recall harm")
        ax.text(
            0.04,
            0.96,
            rf"$n=30$" + "\n" + rf"$\rho={rho:.2f}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#DDDDDD", linewidth=0.4),
        )
        panel_label(ax, f"({chr(ord('a') + col)})", x=-0.08 if col else -0.14, y=1.12)

    pair_handles = [
        Line2D(
            [0],
            [0],
            marker=PAIR_MARKER[p],
            color="none",
            markerfacecolor=PAIR_COLOR[p],
            markeredgecolor=PAIR_COLOR[p],
            markersize=5.5,
            label=PAIR_LABEL[p],
        )
        for p in PAIR_ORDER
    ]
    alg_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#444444", markeredgecolor="#444444", markersize=5.5, label="FedAvg"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#444444", markersize=5.5, label="FedProx"),
    ]
    fig.legend(
        handles=pair_handles + alg_handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.55, 1.02),
        fontsize=7,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(wspace=0.12, left=0.08, right=0.99, top=0.80, bottom=0.20)
    save(fig, "fig3_monopoly_harm")


def _pair_means_from_by_config(rows: list[dict], drop_key: str, harm_key: str) -> dict:
    out: dict[tuple[str, str, float, str], tuple[float, float]] = {}
    for row in rows:
        frac = float(row.get("flip_fraction", 1.0))
        out[(row["alpha_label"], row["algorithm"], frac, row["pair"])] = (
            row[drop_key]["mean"] * 100.0,
            row[harm_key]["mean"] * 100.0,
        )
    return out


def fig4_visibility() -> None:
    attack = load_json(RUNS / "pv19-capped-attack" / "phase4_test_summary.json")
    sens = load_json(RUNS / "pv19-capped-flip-sensitivity" / "phase4_flip_sensitivity_test_summary.json")
    points = _pair_means_from_by_config(
        attack["by_config"],
        "accuracy_drop_vs_clean_e1",
        "source_recall_harm_vs_clean_e1",
    )
    points.update(
        _pair_means_from_by_config(
            sens["by_config"],
            "accuracy_drop_vs_clean_e1",
            "source_recall_harm_vs_clean_e1",
        )
    )
    fractions = (0.25, 0.50, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), sharey=True)
    width = 0.34
    for col, alpha in enumerate(ALPHA_ORDER):
        ax = axes[col]
        x = np.arange(len(fractions))
        drop_means, harm_means = [], []
        for i, frac in enumerate(fractions):
            drops = [points[(alpha, "fedavg", frac, pair)][0] for pair in PAIR_ORDER]
            harms = [points[(alpha, "fedavg", frac, pair)][1] for pair in PAIR_ORDER]
            drop_means.append(float(np.mean(drops)))
            harm_means.append(float(np.mean(harms)))
            for pair, d, h in zip(PAIR_ORDER, drops, harms):
                ax.scatter(i - width / 2, d, s=18, marker=PAIR_MARKER[pair], color=PAIR_COLOR[pair], zorder=4, edgecolors="white", linewidths=0.3)
                ax.scatter(i + width / 2, h, s=18, marker=PAIR_MARKER[pair], color=PAIR_COLOR[pair], zorder=4, edgecolors="white", linewidths=0.3)
        ax.bar(x - width / 2, drop_means, width, color=C["sky"], edgecolor="none", label="Accuracy drop")
        ax.bar(x + width / 2, harm_means, width, color=C["vermillion"], edgecolor="none", label="Source-recall harm")
        ax.set_xticks(x)
        ax.set_xticklabels(["0.25", "0.50", "1.00"])
        ax.set_title(ALPHA_LABEL[alpha])
        ax.set_xlabel("Flip fraction")
        if col == 0:
            ax.set_ylabel("Drop versus paired clean (pp)")
        ax.set_ylim(0, 78)
        panel_label(ax, f"({chr(ord('a') + col)})", x=-0.08 if col else -0.12, y=1.12)
    pair_handles = [
        Line2D(
            [0],
            [0],
            marker=PAIR_MARKER[p],
            color="none",
            markerfacecolor=PAIR_COLOR[p],
            markeredgecolor=PAIR_COLOR[p],
            markersize=5.5,
            label=PAIR_LABEL[p],
        )
        for p in PAIR_ORDER
    ]
    bar_handles = [
        Rectangle((0, 0), 1, 1, color=C["sky"], label="Accuracy drop"),
        Rectangle((0, 0), 1, 1, color=C["vermillion"], label="Source-recall harm"),
    ]
    fig.legend(
        handles=bar_handles + pair_handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.55, 1.02),
        fontsize=7,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    fig.subplots_adjust(wspace=0.12, left=0.08, right=0.99, top=0.80, bottom=0.18)
    save(fig, "fig4_visibility_gap")


def source_recall(folder: str, experiment_id: str, class_name: str) -> float:
    payload = load_json(RUNS / folder / experiment_id / "test_evaluation.json")
    return float(payload["per_class"][class_name]["recall"])


def own_baseline_source_harm() -> dict[tuple[str, str], float]:
    """Pair-averaged attack-induced source harm vs the aggregator's own clean run."""
    defense = load_json(RUNS / "pv19-capped-defense" / "phase4_defense_test_summary.json")
    attack = load_json(RUNS / "pv19-capped-attack" / "phase4_test_summary.json")
    att = {
        (r["alpha_label"], r["algorithm"], r["seed"], r["pair"]): float(r["source_recall"])
        for r in defense["runs"]
    }
    out: dict[tuple[str, str], float] = {}
    for alpha in ALPHA_ORDER:
        fa_harms = [
            row["source_recall_harm_vs_clean_e1"]["mean"]
            for row in attack["by_config"]
            if row["algorithm"] == "fedavg" and row["alpha_label"] == alpha
        ]
        out[(alpha, "fedavg")] = float(np.mean(fa_harms))
        for agg in ("trimmed_mean", "median", "multikrum", "krum"):
            pair_means = []
            for pair, cls in SOURCE_CLASSES.items():
                gaps = []
                for seed in (101, 211, 307, 401, 503):
                    clean = source_recall(
                        "pv19-capped-clean-robust",
                        "pv19-capped-%s-s%s-%s-e1-clean" % (alpha, seed, agg),
                        cls,
                    )
                    gaps.append(clean - att[(alpha, agg, seed, pair)])
                pair_means.append(float(np.mean(gaps)))
            out[(alpha, agg)] = float(np.mean(pair_means))
    return out


def fig5_pareto() -> None:
    clean = load_json(RUNS / "pv19-capped-clean-robust" / "phase4_clean_robust_test_summary.json")
    phase3 = load_json(RUNS / "pv19-capped" / "phase3_test_summary.json")
    harm = own_baseline_source_harm()

    clean_f1 = {(row["alpha_label"], row["algorithm"]): row["macro_f1"]["mean"] for row in clean["by_config"]}
    for row in phase3["by_config"]:
        if row["algorithm"] == "fedavg" and row["local_epochs"] == 1:
            clean_f1[(row["alpha_label"], "fedavg")] = row["macro_f1"]["mean"]

    fig, ax = plt.subplots(figsize=(4.7, 3.45))
    for alpha in ALPHA_ORDER:
        xs, ys = [], []
        for agg in AGG_ORDER:
            x = clean_f1[(alpha, agg)]
            y = harm[(alpha, agg)]
            xs.append(x)
            ys.append(y)
            ax.scatter(
                x,
                y,
                s=52 if agg == "fedavg" else 42,
                marker=AGG_MARKER[agg],
                color=ALPHA_COLOR[alpha],
                edgecolors="white",
                linewidths=0.4,
                zorder=4,
            )
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], color=ALPHA_COLOR[alpha], linewidth=0.8, alpha=0.45, zorder=2)

    ax.annotate(
        "preferred",
        xy=(0.97, 0.03),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=7,
        color=C["green"],
        fontweight="bold",
    )
    ax.annotate(
        "",
        xy=(0.99, 0.015),
        xytext=(0.82, 0.16),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.9),
    )
    ax.set_xlabel("Clean macro-F1 (no attack)")
    ax.set_ylabel("Attack-induced source-recall harm")
    ax.set_xlim(0.12, 0.92)
    ax.set_ylim(0.0, 0.72)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ALPHA_COLOR["a01"], markersize=6, label=r"$\alpha=0.1$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ALPHA_COLOR["a05"], markersize=6, label=r"$\alpha=0.5$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ALPHA_COLOR["iid"], markersize=6, label="IID"),
    ]
    for agg in AGG_ORDER:
        legend_elems.append(
            Line2D(
                [0],
                [0],
                marker=AGG_MARKER[agg],
                color="none",
                markerfacecolor="#444444",
                markeredgecolor="#444444",
                markersize=6,
                label=AGG_LABEL[agg],
            )
        )
    ax.legend(
        handles=legend_elems,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6.5,
        labelspacing=0.28,
        borderaxespad=0.0,
    )
    save(fig, "fig5_aggregator_pareto")


def main() -> None:
    style()
    fig1_testbed()
    fig2_heatmap()
    fig3_scatter()
    fig4_visibility()
    fig5_pareto()


if __name__ == "__main__":
    main()
