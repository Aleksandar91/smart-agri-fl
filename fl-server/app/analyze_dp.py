"""Compare a non-private FedAvg baseline with one or more DP runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("history"), list):
        raise ValueError(f"Missing history list: {path}")
    return payload


def _evaluations(payload: Dict) -> List[Dict]:
    return [
        record for record in payload["history"] if record.get("phase") == "evaluate"
    ]


def _final(payload: Dict, name: str) -> Dict:
    evaluations = _evaluations(payload)
    if not evaluations:
        raise ValueError(f"{name} has no evaluation records")
    return evaluations[-1]


def _difference(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def analyze_dp(baseline: Dict, dp_runs: List[Dict]) -> Dict:
    if not dp_runs:
        raise ValueError("At least one DP history is required")
    base_evals = _evaluations(baseline)
    rounds = [int(record["round"]) for record in base_evals]
    base_final = _final(baseline, "baseline")
    base_acc = base_final.get("global_val_acc")
    base_classes = (base_final.get("per_class_acc") or {}).keys()

    run_summaries = []
    for payload in dp_runs:
        evals = _evaluations(payload)
        run_rounds = [int(record["round"]) for record in evals]
        if run_rounds != rounds:
            raise ValueError(f"Evaluation rounds differ: {rounds} vs {run_rounds}")
        final = _final(payload, payload.get("experiment_id", "dp"))
        classes = (final.get("per_class_acc") or {}).keys()
        if set(classes) != set(base_classes):
            raise ValueError("Per-class metric sets differ")
        per_class = {}
        for class_name in sorted(base_classes):
            dp_value = final["per_class_acc"][class_name]
            base_value = base_final["per_class_acc"][class_name]
            per_class[class_name] = {
                "baseline": base_value,
                "dp": dp_value,
                "delta": _difference(dp_value, base_value),
            }
        ranked = sorted(
            (
                (item["delta"], name)
                for name, item in per_class.items()
                if item["delta"] is not None
            )
        )
        accs = [record.get("global_val_acc") for record in evals]
        peak_round = None
        peak_acc = None
        for record in evals:
            acc = record.get("global_val_acc")
            if acc is None:
                continue
            if peak_acc is None or acc > peak_acc:
                peak_acc = acc
                peak_round = int(record["round"])
        run_summaries.append(
            {
                "experiment_id": payload.get("experiment_id"),
                "dp_config": payload.get("dp_config"),
                "final_accuracy": final.get("global_val_acc"),
                "peak_accuracy": peak_acc,
                "peak_round": peak_round,
                "accuracy_by_round": accs,
                "utility_cost": _difference(final.get("global_val_acc"), base_acc),
                "worst_class_drops": [
                    {
                        "class": name,
                        "delta": delta,
                        "baseline": per_class[name]["baseline"],
                        "dp": per_class[name]["dp"],
                    }
                    for delta, name in ranked[:5]
                ],
                "classes_drop_ge_10pp": sum(
                    1
                    for delta, _name in ranked
                    if delta <= -0.10
                ),
                "per_class_final": per_class,
            }
        )

    return {
        "baseline_experiment_id": baseline.get("experiment_id"),
        "baseline_final_accuracy": base_acc,
        "rounds": rounds,
        "runs": run_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze central DP utility cost.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dp", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = analyze_dp(_load(args.baseline), [_load(path) for path in args.dp])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        "[dp-analysis]",
        {
            "baseline": report["baseline_final_accuracy"],
            "n_runs": len(report["runs"]),
            "output": str(args.output_json),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
