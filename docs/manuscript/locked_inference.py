"""Locked protocol tests and mixed model on archived PV-19-capped test jobs.

Families and coding are listed in
docs/experiment_protocol/locked_inference_plan.md
(31 August 2026; later dated amendments). Not chosen from p-values.
No new FL training. Not a public preregistration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "experiment_protocol"
RUNS = PROTOCOL / "runs"
PART = PROTOCOL / "partitions" / "pv19-capped"
OUT = Path(__file__).resolve().parent / "locked_inference_results.json"

SEEDS = (101, 211, 307, 401, 503)
ALPHAS = ("a01", "a05", "iid")
PAIRS = ("apple", "cherry", "potato")
BOOT_SEED = 20260824
N_BOOT = 10000
ALPHA_LABEL = {"a01": "0.1", "a05": "0.5", "iid": "IID"}
SOURCE = {
    "apple": "Apple___healthy",
    "cherry": "Cherry___healthy",
    "potato": "Potato___healthy",
}
TARGET = {
    "apple": "Apple___Apple_scab",
    "cherry": "Cherry___Powdery_mildew",
    "potato": "Potato___Late_blight",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def to_py(value):
    if isinstance(value, dict):
        return {str(k): to_py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_py(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def holm(pvalues):
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_perm_p(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    observed = float(diffs.mean())
    total = 1 << n
    count = 0
    for mask in range(total):
        acc = 0.0
        for i in range(n):
            acc += diffs[i] if (mask >> i) & 1 else -diffs[i]
        if abs(acc / n) + 1e-15 >= abs(observed):
            count += 1
    return count / float(total), observed


def bootstrap_mean_ci(diffs, rng):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    means = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        means[i] = diffs[rng.randint(0, n, size=n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def wilcoxon_p(diffs):
    diffs = np.asarray(diffs, dtype=float)
    if np.allclose(diffs, 0.0):
        return 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(result.pvalue)
    except ValueError:
        return None


def test_diffs(diffs, family, test_id, comparison, metric, notes="", extra=None):
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.RandomState(BOOT_SEED)
    p_perm, mean_diff = paired_perm_p(diffs)
    lo, hi = bootstrap_mean_ci(diffs, rng)
    sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    dz = float(mean_diff / sd) if sd > 1e-15 else None
    signs = np.sign(np.round(diffs, 12))
    row = {
        "family": family,
        "id": test_id,
        "comparison": comparison,
        "metric": metric,
        "n": int(len(diffs)),
        "mean_diff": mean_diff,
        "sd_diff": sd,
        "cohens_dz": dz,
        "permutation_p": p_perm,
        "wilcoxon_p": wilcoxon_p(diffs),
        "boot_ci95": [lo, hi],
        "n_positive": int(np.sum(diffs > 0)),
        "n_negative": int(np.sum(diffs < 0)),
        "n_zero": int(np.sum(np.abs(diffs) < 1e-15)),
        "all_same_sign": bool(len(set(signs.tolist())) == 1 and not np.any(np.abs(diffs) < 1e-15)),
        "notes": notes,
    }
    if extra:
        row.update(extra)
    return row


def seed_means(value_by_pair_seed):
    """Average the three locked pairs inside each partition seed.

    The mean of the five seed-means equals the unweighted mean of the 15
    pair×seed jobs. Sign-flipping the five seed-means is the same as flipping
    all three pair gaps inside a seed together.
    """
    means = []
    per_seed = {}
    for seed in SEEDS:
        values = [float(value_by_pair_seed[(pair, seed)]) for pair in PAIRS]
        if len(values) != 3:
            raise ValueError("expected three pairs for seed %s" % seed)
        mean_value = float(np.mean(values))
        means.append(mean_value)
        per_seed[str(seed)] = {
            "pairs": {pair: values[i] for i, pair in enumerate(PAIRS)},
            "mean": mean_value,
        }
    job_mean = float(np.mean([float(v) for v in value_by_pair_seed.values()]))
    if abs(float(np.mean(means)) - job_mean) > 1e-12:
        raise ValueError("seed-mean of pairs does not match the 15-job mean")
    return means, per_seed


def job_level_audit(diffs_15):
    """Non-confirmatory: old n=15 sign-flip that treats pairs as exchangeable."""
    diffs = np.asarray(diffs_15, dtype=float)
    rng = np.random.RandomState(BOOT_SEED)
    p_perm, mean_diff = paired_perm_p(diffs)
    lo, hi = bootstrap_mean_ci(diffs, rng)
    return {
        "n": int(len(diffs)),
        "mean_diff": mean_diff,
        "permutation_p": p_perm,
        "boot_ci95": [lo, hi],
        "note": (
            "Not confirmatory. Treats three source pairs that share a partition "
            "and a clean reference as independently sign-flippable."
        ),
    }


SEED_BLOCK_NOTE = (
    "Confirmatory unit is the partition seed (n=5): the three locked pairs "
    "are averaged within seed, then signs are flipped at the seed level. "
    "A percentile bootstrap on five same-sign gaps does not rescue the "
    "exact test, whose two-sided floor is 0.0625."
)
SEED_UNIT_EXTRA = {
    "inference_unit": "partition-seed mean of three locked pairs",
    "n_jobs": 15,
}


def apply_holm(rows):
    pvals = [row["permutation_p"] for row in rows]
    adjusted = holm(pvals)
    for row, p_adj in zip(rows, adjusted):
        row["holm_p"] = p_adj
        row["reject_0.05"] = bool(p_adj < 0.05)
    return rows


def index_runs(runs, extra_keys):
    table = {}
    for run in runs:
        key = tuple(run[k] for k in extra_keys)
        table[key] = run
    return table


def class_metrics(alpha, seed):
    summary = load_json(PART / alpha / ("seed-%s" % seed) / "partitions_summary.json")
    if "class_metrics" in summary:
        return summary["class_metrics"]
    return summary["heterogeneity"]["class_metrics"]


def evaluation_path(folder, experiment_id):
    return RUNS / folder / experiment_id / "test_evaluation.json"


def load_per_class(folder, experiment_id):
    payload = load_json(evaluation_path(folder, experiment_id))
    rows = []
    for class_name, stats in payload["per_class"].items():
        support = int(stats["support"])
        recall = float(stats["recall"])
        n_correct = int(round(recall * support)) if support else 0
        if support:
            n_correct = min(support, max(0, n_correct))
        rows.append(
            {
                "class_name": class_name,
                "recall": recall,
                "support": support,
                "n_correct": n_correct,
            }
        )
    return rows


def family_f1(phase3):
    by_seed = index_runs(
        [r for r in phase3["runs"] if r["local_epochs"] == 1],
        ("alpha_label", "algorithm", "seed"),
    )
    rows = []
    for alpha in ALPHAS:
        diffs = [
            by_seed[(alpha, "fedprox", seed)]["macro_f1"]
            - by_seed[(alpha, "fedavg", seed)]["macro_f1"]
            for seed in SEEDS
        ]
        rows.append(
            test_diffs(
                diffs,
                "F1",
                "F1.%d" % (1 + ALPHAS.index(alpha)),
                "FedProx - FedAvg, alpha=%s, E=1, clean" % ALPHA_LABEL[alpha],
                "macro-F1",
                extra={
                    "inference_unit": "partition seed (one clean job per seed)",
                    "n_jobs": 5,
                },
            )
        )
    return apply_holm(rows)


def family_f1b(phase3):
    by_seed = index_runs(
        [
            r
            for r in phase3["runs"]
            if r["local_epochs"] == 1 and r["algorithm"] == "fedavg"
        ],
        ("alpha_label", "seed"),
    )
    rows = []
    for i, alpha in enumerate(("a01", "a05")):
        diffs = [
            by_seed[(alpha, seed)]["macro_f1"] - by_seed[("iid", seed)]["macro_f1"]
            for seed in SEEDS
        ]
        rows.append(
            test_diffs(
                diffs,
                "F1b",
                "F1b.%d" % (i + 1),
                "FedAvg E=1 clean, alpha=%s - IID" % ALPHA_LABEL[alpha],
                "macro-F1",
                notes="Paired by seed list, not by identical image assignment.",
                extra={
                    "inference_unit": "partition seed (one clean job per seed)",
                    "n_jobs": 5,
                },
            )
        )
    return apply_holm(rows)


def family_f2a(attack):
    by_key = index_runs(
        [r for r in attack["runs"] if r["algorithm"] == "fedavg"],
        ("alpha_label", "pair", "seed"),
    )
    rows = []
    for i, alpha in enumerate(ALPHAS):
        value_by_pair_seed = {
            (pair, seed): by_key[(alpha, pair, seed)]["source_recall_harm"]
            for pair in PAIRS
            for seed in SEEDS
        }
        diffs, per_seed = seed_means(value_by_pair_seed)
        job_diffs = [value_by_pair_seed[(pair, seed)] for pair in PAIRS for seed in SEEDS]
        extra = dict(SEED_UNIT_EXTRA)
        extra["seed_means"] = per_seed
        extra["legacy_job_level"] = job_level_audit(job_diffs)
        rows.append(
            test_diffs(
                diffs,
                "F2a",
                "F2a.%d" % (i + 1),
                "FedAvg flip 1.0 source harm vs 0, alpha=%s" % ALPHA_LABEL[alpha],
                "source-recall harm",
                notes=SEED_BLOCK_NOTE,
                extra=extra,
            )
        )
    return apply_holm(rows)


def family_f2b(attack):
    by_key = index_runs(
        [r for r in attack["runs"] if r["algorithm"] == "fedavg"],
        ("alpha_label", "pair", "seed"),
    )
    rows = []
    for i, alpha in enumerate(ALPHAS):
        value_by_pair_seed = {
            (pair, seed): (
                by_key[(alpha, pair, seed)]["source_recall_harm"]
                - by_key[(alpha, pair, seed)]["accuracy_drop"]
            )
            for pair in PAIRS
            for seed in SEEDS
        }
        diffs, per_seed = seed_means(value_by_pair_seed)
        job_diffs = [value_by_pair_seed[(pair, seed)] for pair in PAIRS for seed in SEEDS]
        extra = dict(SEED_UNIT_EXTRA)
        extra["seed_means"] = per_seed
        extra["legacy_job_level"] = job_level_audit(job_diffs)
        rows.append(
            test_diffs(
                diffs,
                "F2b",
                "F2b.%d" % (i + 1),
                "FedAvg flip 1.0 visibility gap (harm - acc drop), alpha=%s"
                % ALPHA_LABEL[alpha],
                "harm minus accuracy drop",
                notes=SEED_BLOCK_NOTE,
                extra=extra,
            )
        )
    return apply_holm(rows)


def family_f2c(attack):
    by_key = index_runs(
        attack["runs"],
        ("alpha_label", "algorithm", "pair", "seed"),
    )
    rows = []
    for i, alpha in enumerate(ALPHAS):
        value_by_pair_seed = {
            (pair, seed): (
                by_key[(alpha, "fedprox", pair, seed)]["source_recall_harm"]
                - by_key[(alpha, "fedavg", pair, seed)]["source_recall_harm"]
            )
            for pair in PAIRS
            for seed in SEEDS
        }
        diffs, per_seed = seed_means(value_by_pair_seed)
        job_diffs = [value_by_pair_seed[(pair, seed)] for pair in PAIRS for seed in SEEDS]
        extra = dict(SEED_UNIT_EXTRA)
        extra["seed_means"] = per_seed
        extra["legacy_job_level"] = job_level_audit(job_diffs)
        rows.append(
            test_diffs(
                diffs,
                "F2c",
                "F2c.%d" % (i + 1),
                "FedProx - FedAvg source harm, flip 1.0, alpha=%s" % ALPHA_LABEL[alpha],
                "source-recall harm",
                notes=SEED_BLOCK_NOTE,
                extra=extra,
            )
        )
    return apply_holm(rows)


def family_f3(defense, phase3):
    rows = []
    aggregators = (
        ("median", "F3.1"),
        ("trimmed_mean", "F3.2"),
        ("multikrum", "F3.3"),
        ("krum", "F3.4"),
    )
    by_key = index_runs(
        [r for r in defense["runs"] if r["alpha_label"] == "a01"],
        ("algorithm", "pair", "seed"),
    )
    for name, test_id in aggregators:
        value_by_pair_seed = {
            (pair, seed): by_key[(name, pair, seed)][
                "source_recall_recovery_vs_fedavg_attack"
            ]
            for pair in PAIRS
            for seed in SEEDS
        }
        diffs, per_seed = seed_means(value_by_pair_seed)
        job_diffs = [value_by_pair_seed[(pair, seed)] for pair in PAIRS for seed in SEEDS]
        extra = dict(SEED_UNIT_EXTRA)
        extra["seed_means"] = per_seed
        extra["legacy_job_level"] = job_level_audit(job_diffs)
        rows.append(
            test_diffs(
                diffs,
                "F3",
                test_id,
                "%s - attacked FedAvg source-recall recovery, alpha=0.1" % name,
                "source-recall recovery",
                notes=SEED_BLOCK_NOTE,
                extra=extra,
            )
        )

    fedavg = index_runs(
        [
            r
            for r in phase3["runs"]
            if r["local_epochs"] == 1
            and r["algorithm"] == "fedavg"
            and r["alpha_label"] == "a01"
        ],
        ("seed",),
    )
    krum_diffs = []
    for seed in SEEDS:
        krum = load_json(
            evaluation_path(
                "pv19-capped-clean-robust",
                "pv19-capped-a01-s%s-krum-e1-clean" % seed,
            )
        )
        krum_diffs.append(krum["metrics"]["macro_f1"] - fedavg[(seed,)]["macro_f1"])
    rows.append(
        test_diffs(
            krum_diffs,
            "F3",
            "F3.5",
            "clean Krum - clean FedAvg, alpha=0.1, E=1",
            "macro-F1",
            extra={
                "inference_unit": "partition seed (one clean job per seed)",
                "n_jobs": 5,
            },
        )
    )
    return apply_holm(rows)


def spearman_fedavg(attack):
    xs, ys = [], []
    by_seed = {seed: {"x": [], "y": []} for seed in SEEDS}
    for run in attack["runs"]:
        if run["algorithm"] != "fedavg":
            continue
        monopoly = class_metrics(run["alpha_label"], run["seed"])[run["source_class"]][
            "monopoly"
        ]
        harm = run["source_recall_harm"]
        xs.append(monopoly)
        ys.append(harm)
        by_seed[run["seed"]]["x"].append(monopoly)
        by_seed[run["seed"]]["y"].append(harm)

    def rho(a, b):
        ra = np.argsort(np.argsort(np.asarray(a, dtype=float)))
        rb = np.argsort(np.argsort(np.asarray(b, dtype=float)))
        return float(np.corrcoef(ra, rb)[0, 1])

    point = rho(xs, ys)
    rng = np.random.RandomState(BOOT_SEED)
    boot = np.empty(N_BOOT, dtype=float)
    seed_list = list(SEEDS)
    for i in range(N_BOOT):
        drawn = rng.choice(seed_list, size=len(seed_list), replace=True)
        bx, by = [], []
        for seed in drawn:
            bx.extend(by_seed[seed]["x"])
            by.extend(by_seed[seed]["y"])
        boot[i] = rho(bx, by)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_jobs": len(xs),
        "rho": point,
        "seed_clustered_boot_ci95": [float(lo), float(hi)],
        "note": "Descriptive association on 45 FedAvg jobs; CI resamples the five seeds.",
    }


def spearman_xy(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def expected_flip_count(attacker_n, fraction):
    """Match fl-client/app/attacks.py:select_label_flip_positions."""
    if attacker_n <= 0:
        return 0
    if fraction >= 1.0:
        return int(attacker_n)
    count = max(1, int(math.floor(attacker_n * fraction)))
    return min(count, int(attacker_n))


def attacker_flipped_from_history(folder, experiment_id):
    history = load_json(RUNS / folder / experiment_id / "fl_history.json")
    flipped = 0
    for record in history["history"]:
        if record.get("phase") != "fit" or int(record.get("round", 0)) != 1:
            continue
        for client in record.get("client_metrics", []):
            flipped = max(flipped, int(client.get("label_flipped_samples") or 0))
        break
    return flipped


def source_recall_from_eval(folder, experiment_id, source_class):
    payload = load_json(evaluation_path(folder, experiment_id))
    return float(payload["per_class"][source_class]["recall"])


ROBUST_AGGS = ("krum", "median", "multikrum", "trimmed_mean")


def aggregator_source_decomposition(defense):
    """Table 4 harm vs clean FedAvg = clean source cost + own-baseline harm."""
    att = {}
    for run in defense["runs"]:
        att[(run["alpha_label"], run["algorithm"], run["seed"], run["pair"])] = float(
            run["source_recall"]
        )
    rows = []
    for alpha in ALPHAS:
        for agg in ROBUST_AGGS:
            pair_own, pair_fa, pair_cost, pair_clean, pair_att = [], [], [], [], []
            for pair in PAIRS:
                owns, fas, costs, cleans, atts = [], [], [], [], []
                for seed in SEEDS:
                    clean_a = source_recall_from_eval(
                        "pv19-capped-clean-robust",
                        "pv19-capped-%s-s%s-%s-e1-clean" % (alpha, seed, agg),
                        SOURCE[pair],
                    )
                    attacked = att[(alpha, agg, seed, pair)]
                    clean_fa = source_recall_from_eval(
                        "pv19-capped",
                        "pv19-capped-%s-s%s-fedavg-e1" % (alpha, seed),
                        SOURCE[pair],
                    )
                    owns.append(clean_a - attacked)
                    fas.append(clean_fa - attacked)
                    costs.append(clean_fa - clean_a)
                    cleans.append(clean_a)
                    atts.append(attacked)
                pair_own.append(float(np.mean(owns)))
                pair_fa.append(float(np.mean(fas)))
                pair_cost.append(float(np.mean(costs)))
                pair_clean.append(float(np.mean(cleans)))
                pair_att.append(float(np.mean(atts)))
            own = float(np.mean(pair_own))
            fa_harm = float(np.mean(pair_fa))
            cost = float(np.mean(pair_cost))
            if abs((cost + own) - fa_harm) > 1e-9:
                raise ValueError("decomposition failed for %s %s" % (alpha, agg))
            rows.append(
                {
                    "alpha": ALPHA_LABEL[alpha],
                    "aggregator": agg,
                    "clean_source_recall": float(np.mean(pair_clean)),
                    "attacked_source_recall": float(np.mean(pair_att)),
                    "own_baseline_harm": own,
                    "harm_vs_clean_fedavg": fa_harm,
                    "clean_source_cost_vs_fedavg": cost,
                    "pair_means": {
                        pair: {
                            "own_baseline_harm": pair_own[i],
                            "harm_vs_clean_fedavg": pair_fa[i],
                        }
                        for i, pair in enumerate(PAIRS)
                    },
                }
            )
    return {
        "note": (
            "Secondary. Not a Holm family. Pair-averaged like Table 4. "
            "harm_vs_clean_fedavg = clean_source_cost + own_baseline_harm."
        ),
        "rows": rows,
    }


def _confusion_rate(payload, true_cls, pred_cls):
    classes = payload["classes"]
    i = classes.index(true_cls)
    j = classes.index(pred_cls)
    row = payload["confusion_matrix"][i]
    total = float(sum(row))
    return float(row[j] / total) if total else 0.0


def _source_error_to_target(payload, source_cls, target_cls):
    classes = payload["classes"]
    i = classes.index(source_cls)
    j = classes.index(target_cls)
    row = payload["confusion_matrix"][i]
    support = float(sum(row))
    correct = float(row[i])
    to_target = float(row[j])
    errors = support - correct
    frac = float(to_target / errors) if errors else None
    return to_target / support if support else 0.0, frac


def visibility_gap_decomposition(phase3, attack):
    """Accuracy drop = pi_s * H_s + collateral. Also source-to-target confusion."""
    fedavg_clean = index_runs(
        [r for r in phase3["runs"] if r["local_epochs"] == 1 and r["algorithm"] == "fedavg"],
        ("alpha_label", "seed"),
    )
    probe = load_json(
        evaluation_path("pv19-capped", fedavg_clean[("a01", 101)]["experiment_id"])
    )
    n_test = float(probe["num_images"])
    supports = {
        name: int(stats["support"]) for name, stats in probe["per_class"].items()
    }
    pi = {name: supports[name] / n_test for name in supports}
    pair_support = {
        pair: {
            "source_class": SOURCE[pair],
            "target_class": TARGET[pair],
            "source_support": supports[SOURCE[pair]],
            "target_support": supports[TARGET[pair]],
            "pi_source": pi[SOURCE[pair]],
            "pi_target": pi[TARGET[pair]],
        }
        for pair in PAIRS
    }

    job_rows = []
    for run in attack["runs"]:
        if run["algorithm"] != "fedavg":
            continue
        pair = run["pair"]
        source = run["source_class"]
        target = run["target_class"]
        clean_run = fedavg_clean[(run["alpha_label"], run["seed"])]
        clean = load_json(evaluation_path("pv19-capped", clean_run["experiment_id"]))
        att = load_json(evaluation_path("pv19-capped-attack", run["experiment_id"]))
        acc_drop = float(clean["metrics"]["accuracy"] - att["metrics"]["accuracy"])
        recon = 0.0
        for name in probe["per_class"]:
            d_c = float(clean["per_class"][name]["recall"] - att["per_class"][name]["recall"])
            recon += pi[name] * d_c
        h_s = float(clean["per_class"][source]["recall"] - att["per_class"][source]["recall"])
        src_contrib = pi[source] * h_s
        collateral = acc_drop - src_contrib
        p_t_given_s = _confusion_rate(att, source, target)
        p_t_given_s_clean = _confusion_rate(clean, source, target)
        _rate, frac_err = _source_error_to_target(att, source, target)
        job_rows.append(
            {
                "alpha": ALPHA_LABEL[run["alpha_label"]],
                "pair": pair,
                "seed": int(run["seed"]),
                "H_s": h_s,
                "acc_drop": acc_drop,
                "recon_acc_drop": recon,
                "source_contrib_piHs": src_contrib,
                "collateral": collateral,
                "p_target_given_source": p_t_given_s,
                "p_target_given_source_clean": p_t_given_s_clean,
                "frac_source_errors_to_target": frac_err,
                "target_recall_delta": float(run["target_recall_delta"]),
                "target_precision_delta": float(
                    att["per_class"][target]["precision"]
                    - clean["per_class"][target]["precision"]
                ),
                "ba_drop": float(
                    clean["metrics"]["balanced_accuracy"]
                    - att["metrics"]["balanced_accuracy"]
                ),
                "H_s_over_K": h_s / 19.0,
            }
        )
        if abs(acc_drop - recon) > 1e-9:
            raise ValueError("accuracy decomposition failed for %s" % run["experiment_id"])

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    summaries = []
    for alpha in ("0.1", "0.5", "IID"):
        for pair in PAIRS:
            sl = [r for r in job_rows if r["alpha"] == alpha and r["pair"] == pair]
            summaries.append(
                {
                    "alpha": alpha,
                    "pair": pair,
                    "n": len(sl),
                    "H_s": _mean([r["H_s"] for r in sl]),
                    "acc_drop": _mean([r["acc_drop"] for r in sl]),
                    "source_contrib_piHs": _mean([r["source_contrib_piHs"] for r in sl]),
                    "collateral": _mean([r["collateral"] for r in sl]),
                    "p_target_given_source": _mean(
                        [r["p_target_given_source"] for r in sl]
                    ),
                    "frac_source_errors_to_target": _mean(
                        [r["frac_source_errors_to_target"] for r in sl]
                    ),
                    "target_recall_delta": _mean([r["target_recall_delta"] for r in sl]),
                    "target_precision_delta": _mean(
                        [r["target_precision_delta"] for r in sl]
                    ),
                    "ba_drop": _mean([r["ba_drop"] for r in sl]),
                    "H_s_over_K": _mean([r["H_s_over_K"] for r in sl]),
                }
            )
        pair_means = [s for s in summaries if s["alpha"] == alpha and s["pair"] != "unweighted_pair_mean"]
        summaries.append(
            {
                "alpha": alpha,
                "pair": "unweighted_pair_mean",
                "n": 15,
                "H_s": _mean([s["H_s"] for s in pair_means]),
                "acc_drop": _mean([s["acc_drop"] for s in pair_means]),
                "source_contrib_piHs": _mean(
                    [s["source_contrib_piHs"] for s in pair_means]
                ),
                "collateral": _mean([s["collateral"] for s in pair_means]),
                "p_target_given_source": _mean(
                    [s["p_target_given_source"] for s in pair_means]
                ),
                "frac_source_errors_to_target": _mean(
                    [s["frac_source_errors_to_target"] for s in pair_means]
                ),
                "target_recall_delta": _mean(
                    [s["target_recall_delta"] for s in pair_means]
                ),
                "target_precision_delta": _mean(
                    [s["target_precision_delta"] for s in pair_means]
                ),
                "ba_drop": _mean([s["ba_drop"] for s in pair_means]),
                "H_s_over_K": _mean([s["H_s_over_K"] for s in pair_means]),
            }
        )

    return {
        "note": (
            "Secondary. Not a Holm family. Accuracy drop equals the support-weighted "
            "sum of class recall changes. F2b (H_s minus accuracy drop) is kept as an "
            "operational visibility summary, not as a scientific null that those two "
            "scales should match."
        ),
        "n_test": int(n_test),
        "n_classes": 19,
        "pair_support": pair_support,
        "summaries": summaries,
    }


def ols_fit_payload(formula, frame):
    fit = smf.ols(formula, frame).fit()
    terms = []
    ci = fit.conf_int()
    for name in fit.params.index:
        terms.append(
            {
                "term": str(name),
                "estimate": float(fit.params[name]),
                "se": float(fit.bse[name]),
                "p": float(fit.pvalues[name]),
                "ci95": [float(ci.loc[name, 0]), float(ci.loc[name, 1])],
            }
        )
    return {
        "formula": formula,
        "n": int(len(frame)),
        "r2": float(fit.rsquared),
        "r2_adj": float(fit.rsquared_adj),
        "coefficients": terms,
    }


def loso_rmse(formula, frame):
    squares = []
    n_held = 0
    for seed in SEEDS:
        train = frame[frame["seed"] != seed]
        test = frame[frame["seed"] == seed]
        fit = smf.ols(formula, train).fit()
        pred = fit.predict(test)
        err = np.asarray(test["harm"], dtype=float) - np.asarray(pred, dtype=float)
        squares.append(float(np.mean(np.square(err))))
        n_held += int(len(test))
    return {
        "formula": formula,
        "n_jobs": int(len(frame)),
        "n_held_out_jobs": n_held,
        "rmse": float(math.sqrt(float(np.mean(squares)))),
    }


def build_flip1_harm_frame(attack):
    records = []
    for run in attack["runs"]:
        metrics = class_metrics(run["alpha_label"], run["seed"])[run["source_class"]]
        counts = [int(x) for x in metrics["counts"]]
        n_c = int(sum(counts))
        attacker_n = int(max(counts))
        monopoly = float(metrics["monopoly"])
        flipped = attacker_flipped_from_history(
            "pv19-capped-attack", run["experiment_id"]
        )
        if flipped != attacker_n:
            raise ValueError(
                "flip-1.0 flipped count %s != attacker source n %s for %s"
                % (flipped, attacker_n, run["experiment_id"])
            )
        realized = flipped / float(n_c) if n_c else None
        if realized is None or abs(realized - monopoly) > 1e-12:
            raise ValueError("flip-1.0 realized fraction must equal Mc")
        records.append(
            {
                "alpha": run["alpha_label"],
                "seed": int(run["seed"]),
                "algorithm": run["algorithm"],
                "pair": run["pair"],
                "Mc": monopoly,
                "entropy": float(metrics["normalized_entropy"]),
                "N_c": n_c,
                "attacker_n": attacker_n,
                "q": 1.0,
                "qMc": monopoly,
                "realized_frac": realized,
                "flipped": flipped,
                "harm": float(run["source_recall_harm"]),
            }
        )
    return pd.DataFrame.from_records(records)


def build_dose_harm_frame(attack, phase3):
    clean_id = {}
    for run in phase3["runs"]:
        if run["local_epochs"] != 1:
            continue
        clean_id[(run["alpha_label"], run["algorithm"], int(run["seed"]))] = run[
            "experiment_id"
        ]
    records = build_flip1_harm_frame(attack).to_dict("records")
    tags = {0.25: "025", 0.50: "050"}
    for alpha in ALPHAS:
        for seed in SEEDS:
            for algorithm in ("fedavg", "fedprox"):
                for pair in PAIRS:
                    source = SOURCE[pair]
                    metrics = class_metrics(alpha, seed)[source]
                    counts = [int(x) for x in metrics["counts"]]
                    n_c = int(sum(counts))
                    attacker_n = int(max(counts))
                    monopoly = float(metrics["monopoly"])
                    clean_recall = source_recall_from_eval(
                        "pv19-capped",
                        clean_id[(alpha, algorithm, seed)],
                        source,
                    )
                    for fraction, tag in tags.items():
                        experiment_id = (
                            "pv19-capped-%s-s%s-%s-e1-flip%s-%s"
                            % (alpha, seed, algorithm, tag, pair)
                        )
                        flipped = attacker_flipped_from_history(
                            "pv19-capped-flip-sensitivity", experiment_id
                        )
                        expected = expected_flip_count(attacker_n, fraction)
                        if flipped != expected:
                            raise ValueError(
                                "realized flip %s != expected %s for %s"
                                % (flipped, expected, experiment_id)
                            )
                        attacked_recall = source_recall_from_eval(
                            "pv19-capped-flip-sensitivity",
                            experiment_id,
                            source,
                        )
                        records.append(
                            {
                                "alpha": alpha,
                                "seed": int(seed),
                                "algorithm": algorithm,
                                "pair": pair,
                                "Mc": monopoly,
                                "entropy": float(metrics["normalized_entropy"]),
                                "N_c": n_c,
                                "attacker_n": attacker_n,
                                "q": float(fraction),
                                "qMc": float(fraction) * monopoly,
                                "realized_frac": flipped / float(n_c),
                                "flipped": flipped,
                                "harm": float(clean_recall - attacked_recall),
                            }
                        )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != 270:
        raise ValueError("dose frame should have 270 jobs, got %s" % len(frame))
    return frame


def rq2_alpha_and_dose(attack, phase3):
    """Secondary analysis (8 Sep 2026): monopoly vs α and poison dose qMc.

    Not a new confirmatory family. Table 10 is unchanged.
    """
    flip1 = build_flip1_harm_frame(attack)
    dose = build_dose_harm_frame(attack, phase3)
    q_lt1 = dose[dose["q"] < 1.0]
    alpha_dummies = pd.get_dummies(flip1["alpha"], drop_first=True).astype(float)

    def residual(series, dummies):
        work = pd.DataFrame({"y": np.asarray(series, dtype=float)})
        for col in dummies.columns:
            work[str(col)] = dummies[col].values
        fit = smf.ols("y ~ " + " + ".join(str(c) for c in dummies.columns), work).fit()
        return np.asarray(fit.resid, dtype=float)

    within = {}
    for alpha in ALPHAS:
        sub = flip1[flip1["alpha"] == alpha]
        fed = sub[sub["algorithm"] == "fedavg"]
        within[alpha] = {
            "n_all": int(len(sub)),
            "rho_all": spearman_xy(sub["Mc"], sub["harm"]),
            "n_fedavg": int(len(fed)),
            "rho_fedavg": spearman_xy(fed["Mc"], fed["harm"]),
            "Mc_min": float(sub["Mc"].min()),
            "Mc_max": float(sub["Mc"].max()),
        }
    dose_within = {}
    for alpha in ALPHAS:
        sub = dose[dose["alpha"] == alpha]
        dose_within[alpha] = {
            "n": int(len(sub)),
            "rho_qMc": spearman_xy(sub["qMc"], sub["harm"]),
            "rho_Mc": spearman_xy(sub["Mc"], sub["harm"]),
            "rho_realized": spearman_xy(sub["realized_frac"], sub["harm"]),
        }
    flip1_models = [
        ols_fit_payload("harm ~ C(alpha)", flip1),
        ols_fit_payload("harm ~ C(alpha) + Mc", flip1),
        ols_fit_payload("harm ~ C(alpha) + entropy", flip1),
        ols_fit_payload("harm ~ C(alpha) + C(pair) + C(algorithm) + Mc", flip1),
    ]
    dose_models = [
        ols_fit_payload("harm ~ C(alpha)", dose),
        ols_fit_payload("harm ~ C(alpha) + Mc", dose),
        ols_fit_payload("harm ~ C(alpha) + q", dose),
        ols_fit_payload("harm ~ C(alpha) + qMc", dose),
        ols_fit_payload("harm ~ C(alpha) + qMc + Mc", dose),
    ]
    return {
        "role": (
            "Secondary analysis after reviews: does monopoly add prediction "
            "beyond α, and is that association distinct from poison dose qMc?"
        ),
        "flip1_n": int(len(flip1)),
        "dose_n": int(len(dose)),
        "flip1_Mc_equals_realized_max_abs_diff": float(
            (flip1["Mc"] - flip1["realized_frac"]).abs().max()
        ),
        "q_lt1_realized_vs_qMc_max_abs_diff": float(
            (q_lt1["realized_frac"] - q_lt1["qMc"]).abs().max()
        ),
        "spearman_flip1": {
            "all_90": spearman_xy(flip1["Mc"], flip1["harm"]),
            "fedavg_45": spearman_xy(
                flip1[flip1["algorithm"] == "fedavg"]["Mc"],
                flip1[flip1["algorithm"] == "fedavg"]["harm"],
            ),
            "partial_given_alpha_all_90": spearman_xy(
                residual(flip1["Mc"], alpha_dummies),
                residual(flip1["harm"], alpha_dummies),
            ),
            "within_alpha": within,
        },
        "spearman_dose_270": {
            "qMc": spearman_xy(dose["qMc"], dose["harm"]),
            "realized": spearman_xy(dose["realized_frac"], dose["harm"]),
            "Mc": spearman_xy(dose["Mc"], dose["harm"]),
            "within_alpha": dose_within,
        },
        "ols_flip1": flip1_models,
        "ols_dose": dose_models,
        "loso_flip1": [
            loso_rmse("harm ~ C(alpha)", flip1),
            loso_rmse("harm ~ C(alpha) + Mc", flip1),
            loso_rmse("harm ~ C(alpha) + entropy", flip1),
        ],
        "loso_dose": [
            loso_rmse("harm ~ C(alpha)", dose),
            loso_rmse("harm ~ C(alpha) + q", dose),
            loso_rmse("harm ~ C(alpha) + Mc", dose),
            loso_rmse("harm ~ C(alpha) + qMc", dose),
            loso_rmse("harm ~ C(alpha) + qMc + Mc", dose),
        ],
        "reading": (
            "At flip 1.0, Mc equals the globally corrupted source-class fraction "
            "by the attacker rule. Across fractions 0.25/0.50/1.0, harm tracks "
            "qMc; adding Mc beside qMc does not improve the fit."
        ),
    }


FORMULA = (
    "recall ~ monopoly + entropy + C(algorithm) + targeted "
    "+ monopoly:targeted + C(algorithm):targeted"
)


def mixed_rows():
    phase3 = load_json(RUNS / "pv19-capped" / "phase3_test_summary.json")
    attack = load_json(RUNS / "pv19-capped-attack" / "phase4_test_summary.json")
    records = []

    for run in phase3["runs"]:
        if run["local_epochs"] != 1:
            continue
        metrics = class_metrics(run["alpha_label"], run["seed"])
        for row in load_per_class("pv19-capped", run["experiment_id"]):
            name = row["class_name"]
            records.append(
                {
                    "seed": int(run["seed"]),
                    "algorithm": run["algorithm"],
                    "class_name": name,
                    "recall": row["recall"],
                    "support": row["support"],
                    "n_correct": row["n_correct"],
                    "monopoly": metrics[name]["monopoly"],
                    "entropy": metrics[name]["normalized_entropy"],
                    "targeted": 0,
                    "attack_job": 0,
                    "alpha": run["alpha_label"],
                    "job": run["experiment_id"],
                }
            )

    for run in attack["runs"]:
        metrics = class_metrics(run["alpha_label"], run["seed"])
        source = run["source_class"]
        for row in load_per_class("pv19-capped-attack", run["experiment_id"]):
            name = row["class_name"]
            records.append(
                {
                    "seed": int(run["seed"]),
                    "algorithm": run["algorithm"],
                    "class_name": name,
                    "recall": row["recall"],
                    "support": row["support"],
                    "n_correct": row["n_correct"],
                    "monopoly": metrics[name]["monopoly"],
                    "entropy": metrics[name]["normalized_entropy"],
                    "targeted": int(name == source),
                    "attack_job": 1,
                    "alpha": run["alpha_label"],
                    "job": run["experiment_id"],
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame["algorithm"] = pd.Categorical(
        frame["algorithm"], categories=["fedavg", "fedprox"]
    )
    return frame


def _coef_rows(fit):
    params = fit.fe_params
    bse = fit.bse
    pvalues = fit.pvalues
    ci = fit.conf_int()
    rows = []
    for name in params.index:
        rows.append(
            {
                "term": str(name),
                "estimate": float(params[name]),
                "se": float(bse[name]),
                "p": float(pvalues[name]),
                "ci95": [float(ci.loc[name, 0]), float(ci.loc[name, 1])],
            }
        )
    return rows


def _named_vcomp(fit):
    named = {}
    vals = np.asarray(fit.vcomp).ravel() if hasattr(fit, "vcomp") else np.array([])
    names = None
    if hasattr(fit.model, "vc_names"):
        names = fit.model.vc_names
    if names is None:
        names = ["vc_%s" % i for i in range(len(vals))]
    elif not isinstance(names, (list, tuple)):
        names = list(names)
    for i, value in enumerate(vals):
        key = str(names[i]) if i < len(names) else "vc_%s" % i
        named[key] = float(value)
    return named


def _residual_summary(fit, frame):
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    resid = np.asarray(fit.resid, dtype=float)
    y = np.asarray(frame["recall"], dtype=float)
    return {
        "n": int(len(resid)),
        "resid_mean": float(resid.mean()),
        "resid_sd": float(resid.std(ddof=1)),
        "fitted_min": float(fitted.min()),
        "fitted_max": float(fitted.max()),
        "n_fitted_outside_01": int(((fitted < 0.0) | (fitted > 1.0)).sum()),
        "n_obs_at_0": int((y <= 1e-12).sum()),
        "n_obs_at_1": int((y >= 1.0 - 1e-12).sum()),
    }


def _find_term(coefficients, exact):
    for row in coefficients:
        if row["term"] == exact:
            return row
    return None


def fit_mixedlm(frame, extra_terms="", formula=None, grouping="crossed_seed_class"):
    formula = FORMULA + extra_terms if formula is None else formula
    frame = frame.copy()
    frame["log_support"] = np.log(np.maximum(frame["support"].astype(float), 1.0))
    frame["ones"] = 1
    result = {
        "formula": formula,
        "grouping": grouping,
        "n_rows": int(len(frame)),
        "n_targeted": int(frame["targeted"].sum()),
        "n_jobs": int(frame["job"].nunique()),
        "n_targeted_classes": int(
            frame.loc[frame["targeted"] == 1, "class_name"].nunique()
        ),
        "converged": False,
        "method": None,
        "coefficients": [],
        "random_effects": {},
        "warning": None,
    }
    try:
        if grouping == "crossed_seed_class":
            model = smf.mixedlm(
                formula,
                frame,
                groups=frame["ones"],
                re_formula="~0",
                vc_formula={"seed": "0 + C(seed)", "klass": "0 + C(class_name)"},
            )
            result["method"] = "crossed LMM, (1|seed)+(1|class), REML L-BFGS"
        elif grouping == "job":
            model = smf.mixedlm(formula, frame, groups=frame["job"])
            result["method"] = "LMM, (1|job), REML L-BFGS"
        elif grouping == "seed":
            model = smf.mixedlm(formula, frame, groups=frame["seed"])
            result["method"] = "LMM, (1|seed) only, REML L-BFGS"
        else:
            raise ValueError("unknown grouping %s" % grouping)
        fit = model.fit(method="lbfgs", reml=True, maxiter=400)
        result["converged"] = bool(fit.converged)
        result["variance_components"] = _named_vcomp(fit)
        if grouping == "crossed_seed_class":
            vals = list(result["variance_components"].values())
            named = {}
            if len(vals) >= 1:
                named["seed"] = vals[0]
            if len(vals) >= 2:
                named["klass"] = vals[1]
            if named:
                result["variance_components"] = named
        if hasattr(fit, "scale"):
            result["residual_var"] = float(fit.scale)
        result["coefficients"] = _coef_rows(fit)
        result["residual_summary"] = _residual_summary(fit, frame)
        return result, fit
    except Exception as exc:
        result["warning"] = "%s failed: %s" % (grouping, exc)

    if grouping != "seed":
        try:
            model = smf.mixedlm(formula, frame, groups=frame["seed"])
            fit = model.fit(method="lbfgs", reml=True, maxiter=400)
            result["method"] = "fallback LMM, (1|seed) only"
            result["converged"] = bool(fit.converged)
            result["residual_var"] = float(fit.scale)
            result["coefficients"] = _coef_rows(fit)
            result["residual_summary"] = _residual_summary(fit, frame)
            return result, fit
        except Exception as exc:
            result["warning"] = (result.get("warning") or "") + " | seed LMM failed: %s" % exc
    return result, None


def collinearity_mc_entropy(frame):
    mc = frame["monopoly"].astype(float)
    ent = frame["entropy"].astype(float)
    r = float(mc.corr(ent))
    targeted = frame[frame["targeted"] == 1]
    r_t = float(targeted["monopoly"].corr(targeted["entropy"])) if len(targeted) else None
    return {
        "n_rows": int(len(frame)),
        "pearson_all_rows": r,
        "vif_two_predictor": float(1.0 / (1.0 - r * r)) if abs(r) < 1 else None,
        "n_targeted": int(len(targeted)),
        "pearson_targeted_rows": r_t,
        "reading": (
            "Monopoly and normalised entropy are two readings of the same "
            "five-client share vector; conditional main effects are not separable."
        ),
    }


def lmm_specification_audit(frame, attack):
    """Secondary stress-test of Table 10 (8 Sep 2026). Not a new Holm family."""
    protocol, _ = fit_mixedlm(frame)
    job_dep, _ = fit_mixedlm(frame, grouping="job")
    attack_job, _ = fit_mixedlm(
        frame,
        formula=(
            "recall ~ monopoly + entropy + C(algorithm) + attack_job + targeted "
            "+ monopoly:targeted + C(algorithm):targeted"
        ),
    )
    mc_only, _ = fit_mixedlm(
        frame,
        formula=(
            "recall ~ monopoly + C(algorithm) + targeted "
            "+ monopoly:targeted + C(algorithm):targeted"
        ),
    )
    ent_only, _ = fit_mixedlm(
        frame,
        formula=(
            "recall ~ entropy + C(algorithm) + targeted "
            "+ entropy:targeted + C(algorithm):targeted"
        ),
    )

    harm = build_flip1_harm_frame(attack)
    harm_ols = ols_fit_payload(
        "harm ~ Mc + C(alpha) + C(algorithm) + C(pair)", harm
    )
    try:
        harm_lmm = smf.mixedlm(
            "harm ~ Mc + C(alpha) + C(algorithm) + C(pair)",
            harm,
            groups=harm["seed"],
        ).fit(method="lbfgs", reml=True, maxiter=400)
        harm_lmm_payload = {
            "formula": "harm ~ Mc + C(alpha) + C(algorithm) + C(pair)",
            "method": "LMM, (1|seed), REML L-BFGS",
            "n": int(len(harm)),
            "converged": bool(harm_lmm.converged),
            "residual_var": float(harm_lmm.scale),
            "coefficients": _coef_rows(harm_lmm),
        }
    except Exception as exc:
        harm_lmm_payload = {"warning": str(exc)}

    interaction = _find_term(protocol["coefficients"], "monopoly:targeted")
    job_ix = _find_term(job_dep["coefficients"], "monopoly:targeted")
    aj_ix = _find_term(attack_job["coefficients"], "monopoly:targeted")
    mc_ix = _find_term(mc_only["coefficients"], "monopoly:targeted")
    ent_ix = _find_term(ent_only["coefficients"], "entropy:targeted")
    harm_mc = _find_term(harm_ols["coefficients"], "Mc")
    harm_lmm_mc = None
    if "coefficients" in harm_lmm_payload:
        harm_lmm_mc = _find_term(harm_lmm_payload["coefficients"], "Mc")

    vcs = protocol.get("variance_components") or {}
    class_vc = None
    seed_vc = None
    if isinstance(vcs, dict):
        for key, value in vcs.items():
            low = str(key).lower()
            if "klass" in low or "class" in low:
                class_vc = value
            if "seed" in low:
                seed_vc = value
        vals = list(vcs.values())
        if seed_vc is None and vals:
            seed_vc = vals[0]
        if class_vc is None and len(vals) > 1:
            class_vc = vals[1]

    return {
        "role": (
            "Secondary specification audit of the protocol Gaussian LMM. "
            "Table 10 remains the protocol fit. Not a Holm family."
        ),
        "collinearity": collinearity_mc_entropy(frame),
        "protocol": {
            "converged": protocol.get("converged"),
            "method": protocol.get("method"),
            "variance_components": protocol.get("variance_components"),
            "residual_var": protocol.get("residual_var"),
            "residual_summary": protocol.get("residual_summary"),
            "n_targeted_classes": protocol.get("n_targeted_classes"),
            "monopoly_targeted": interaction,
            "boundary_component": {
                "name": "klass (class intercept)",
                "estimate": class_vc,
                "seed_intercept": seed_vc,
                "note": (
                    "The class intercept sits on the boundary relative to residual "
                    "variance. Seed intercept is interior."
                ),
            },
        },
        "sensitivities": {
            "job_random_intercept": {
                "monopoly_targeted": job_ix,
                "converged": job_dep.get("converged"),
                "method": job_dep.get("method"),
                "residual_var": job_dep.get("residual_var"),
                "warning": job_dep.get("warning"),
            },
            "attack_job_plus_targeted": {
                "monopoly_targeted": aj_ix,
                "attack_job": _find_term(attack_job["coefficients"], "attack_job"),
                "converged": attack_job.get("converged"),
                "warning": attack_job.get("warning"),
            },
            "monopoly_only": {
                "monopoly_targeted": mc_ix,
                "monopoly_main": _find_term(mc_only["coefficients"], "monopoly"),
                "converged": mc_only.get("converged"),
            },
            "entropy_only": {
                "entropy_targeted": ent_ix,
                "entropy_main": _find_term(ent_only["coefficients"], "entropy"),
                "converged": ent_only.get("converged"),
            },
        },
        "paired_source_harm": {
            "n": int(len(harm)),
            "ols": harm_ols,
            "lmm_seed": harm_lmm_payload,
            "reading": (
                "Direct model of paired source-recall harm on the 90 flip-1.0 jobs. "
                "Mc is the poison dose at q=1. Pair fixed effects keep the three "
                "locked sources from being treated as 19 independently attacked classes."
            ),
        },
        "headline": {
            "protocol_monopoly_targeted": interaction,
            "job_grouped_monopoly_targeted": job_ix,
            "attack_job_monopoly_targeted": aj_ix,
            "ols_harm_Mc": harm_mc,
            "lmm_harm_Mc": harm_lmm_mc,
        },
    }


def fit_gee(frame):
    work = frame[frame["support"] > 0].copy()
    work["prop"] = work["n_correct"] / work["support"].astype(float)
    out = {
        "formula": FORMULA,
        "n_rows": int(len(work)),
        "n_clusters": int(work["seed"].nunique()),
        "method": "binomial GEE, independence, clustered by seed, var_weights=support",
        "coefficients": [],
        "warning": "Five seed clusters make the sandwich SE a sensitivity check, not a substitute for the LMM.",
    }
    try:
        model = smf.gee(
            FORMULA,
            groups="seed",
            data=work,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
            var_weights=work["support"].astype(float),
        )
        fit = model.fit()
        ci = fit.conf_int()
        for name in fit.params.index:
            out["coefficients"].append(
                {
                    "term": str(name),
                    "estimate": float(fit.params[name]),
                    "se": float(fit.bse[name]),
                    "p": float(fit.pvalues[name]),
                    "ci95": [float(ci.loc[name, 0]), float(ci.loc[name, 1])],
                }
            )
    except Exception as exc:
        out["warning"] = str(exc)
    return out


def main():
    phase3 = load_json(RUNS / "pv19-capped" / "phase3_test_summary.json")
    attack = load_json(RUNS / "pv19-capped-attack" / "phase4_test_summary.json")
    defense = load_json(RUNS / "pv19-capped-defense" / "phase4_defense_test_summary.json")

    tests = []
    tests.extend(family_f1(phase3))
    tests.extend(family_f1b(phase3))
    tests.extend(family_f2a(attack))
    tests.extend(family_f2b(attack))
    tests.extend(family_f2c(attack))
    tests.extend(family_f3(defense, phase3))

    print("Building class-level frame for mixed model...")
    frame = mixed_rows()
    lmm, _fit = fit_mixedlm(frame)
    lmm_support, _ = fit_mixedlm(frame, extra_terms=" + log_support")
    lmm_support["role"] = "exploratory: protocol LMM plus log test support"
    gee = fit_gee(frame)
    print("Secondary RQ2 analysis: monopoly vs alpha and qMc...")
    rq2 = rq2_alpha_and_dose(attack, phase3)
    print("Secondary LMM specification audit...")
    lmm_audit = lmm_specification_audit(frame, attack)
    print("Secondary aggregator own-baseline source harm...")
    agg_decomp = aggregator_source_decomposition(defense)
    print("Secondary visibility-gap decomposition...")
    vis = visibility_gap_decomposition(phase3, attack)

    payload = {
        "plan": "docs/experiment_protocol/locked_inference_plan.md",
        "plan_amendment": (
            "2026-09-08 (1) F2a/F2b/F2c/F3.1–F3.4 use partition-seed means "
            "(n=5). (2) Secondary α-vs-monopoly and qMc dose analysis. "
            "(3) Secondary LMM specification audit (job intercept, attack_job, "
            "collinearity, paired-harm); Table 10 protocol formula unchanged. "
            "(4) 2026-09-09 aggregator own-baseline source harm (Table 7). "
            "(5) 2026-09-10 visibility-gap arithmetic and source-to-target confusion. "
            "See locked_inference_plan.md."
        ),
        "rng_seed": BOOT_SEED,
        "n_bootstrap": N_BOOT,
        "targeted_coding": (
            "targeted=1 iff the class is the locked label-flip source in that job"
        ),
        "tests": tests,
        "spearman_fedavg": spearman_fedavg(attack),
        "mixedlm": lmm,
        "mixedlm_support_sensitivity": lmm_support,
        "gee_binomial_sensitivity": gee,
        "rq2_alpha_and_dose": rq2,
        "lmm_specification_audit": lmm_audit,
        "aggregator_source_decomposition": agg_decomp,
        "visibility_gap_decomposition": vis,
        "n_class_rows": int(len(frame)),
        "n_jobs_in_lmm": int(frame["job"].nunique()),
    }
    OUT.write_text(json.dumps(to_py(payload), indent=2), encoding="utf-8")
    print("wrote", OUT)

    print("\n=== Confirmatory tests ===")
    header = "{:<7} {:>8} {:>8} {:>10} {:>10} {:>10}  {}".format(
        "id", "n", "mean", "perm p", "Holm p", "rej?", "comparison"
    )
    print(header)
    for row in tests:
        print(
            "{:<7} {:>8d} {:>8.3f} {:>10.4f} {:>10.4f} {:>10}  {}".format(
                row["id"],
                row["n"],
                row["mean_diff"],
                row["permutation_p"],
                row["holm_p"],
                "yes" if row["reject_0.05"] else "no",
                row["comparison"],
            )
        )
    print("\n=== LMM coefficients ===")
    print("method:", lmm.get("method"), "converged:", lmm.get("converged"))
    if lmm.get("warning"):
        print("warning:", lmm["warning"])
    for coef in lmm.get("coefficients", []):
        print(
            "  {:<44} {:+.4f}  se={:.4f}  p={:.4g}".format(
                coef["term"], coef["estimate"], coef["se"], coef["p"]
            )
        )
    print("\nSpearman FedAvg:", payload["spearman_fedavg"])
    print("RQ2 alpha/dose LOSO flip1:", payload["rq2_alpha_and_dose"]["loso_flip1"])
    print("RQ2 alpha/dose LOSO 270:", payload["rq2_alpha_and_dose"]["loso_dose"])
    print("LMM VCs:", lmm.get("variance_components"), "resid", lmm.get("residual_var"))
    print("LMM audit headline:", lmm_audit.get("headline"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
