"""Compare an instrumented clean FL history with a poisoning run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load(path: Path) -> Dict:
    if not path.is_file():
        raise ValueError(f"History not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError(f"Missing history list: {path}")
    return payload


def _phase_records(payload: Dict, phase: str) -> List[Dict]:
    return [record for record in payload["history"] if record.get("phase") == phase]


def _peak(records: List[Dict]) -> Dict:
    valid = [
        record for record in records if record.get("global_val_acc") is not None
    ]
    if not valid:
        return {"round": None, "accuracy": None}
    record = max(valid, key=lambda item: float(item["global_val_acc"]))
    return {
        "round": int(record["round"]),
        "accuracy": float(record["global_val_acc"]),
    }


def _last(records: List[Dict], name: str) -> Dict:
    if not records:
        raise ValueError(f"{name} history contains no evaluation records")
    return records[-1]


def compare_histories(clean: Dict, attacked: Dict) -> Dict:
    clean_eval = _phase_records(clean, "evaluate")
    attack_eval = _phase_records(attacked, "evaluate")
    clean_rounds = [int(record["round"]) for record in clean_eval]
    attack_rounds = [int(record["round"]) for record in attack_eval]
    if clean_rounds != attack_rounds:
        raise ValueError(
            f"Evaluation rounds differ: clean={clean_rounds}, attacked={attack_rounds}"
        )

    clean_last = _last(clean_eval, "Clean")
    attack_last = _last(attack_eval, "Attacked")
    clean_classes = clean_last.get("per_class_acc") or {}
    attack_classes = attack_last.get("per_class_acc") or {}
    if set(clean_classes) != set(attack_classes):
        raise ValueError("Clean and attacked per-class metric sets differ")

    per_class = {}
    for class_name in sorted(clean_classes):
        clean_value = clean_classes[class_name]
        attack_value = attack_classes[class_name]
        per_class[class_name] = {
            "clean": clean_value,
            "attacked": attack_value,
            "delta": (
                float(attack_value) - float(clean_value)
                if clean_value is not None and attack_value is not None
                else None
            ),
            "validation_samples": (attack_last.get("per_class_total") or {}).get(
                class_name
            ),
        }

    attack_audit = []
    for fit_record in _phase_records(attacked, "fit"):
        active_clients = [
            metrics
            for metrics in fit_record.get("client_metrics", [])
            if metrics.get("attack_active")
        ]
        attack_audit.append(
            {
                "round": int(fit_record["round"]),
                "active_clients": active_clients,
            }
        )

    clean_final = clean_last.get("global_val_acc")
    attack_final = attack_last.get("global_val_acc")
    return {
        "clean_experiment_id": clean.get("experiment_id"),
        "attacked_experiment_id": attacked.get("experiment_id"),
        "rounds": clean_rounds,
        "clean_final_accuracy": clean_final,
        "attacked_final_accuracy": attack_final,
        "final_accuracy_delta": (
            float(attack_final) - float(clean_final)
            if clean_final is not None and attack_final is not None
            else None
        ),
        "clean_peak": _peak(clean_eval),
        "attacked_peak": _peak(attack_eval),
        "trajectory": [
            {
                "round": clean_rounds[index],
                "clean_accuracy": clean_eval[index].get("global_val_acc"),
                "attacked_accuracy": attack_eval[index].get("global_val_acc"),
            }
            for index in range(len(clean_rounds))
        ],
        "per_class_final": per_class,
        "attack_audit": attack_audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare clean and poisoned Flower history JSON files."
    )
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--attacked", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = compare_histories(_load(args.clean), _load(args.attacked))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        "[poison-analysis]",
        {
            "clean": report["clean_experiment_id"],
            "attacked": report["attacked_experiment_id"],
            "clean_final_accuracy": report["clean_final_accuracy"],
            "attacked_final_accuracy": report["attacked_final_accuracy"],
            "delta": report["final_accuracy_delta"],
            "output": str(args.output_json),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
