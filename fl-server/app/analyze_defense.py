"""Compare clean/attacked FedAvg with clean/attacked robust aggregation."""

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


def analyze_defense(
    clean_fedavg: Dict,
    attacked_fedavg: Dict,
    clean_defense: Dict,
    attacked_defense: Dict,
) -> Dict:
    payloads = {
        "clean_fedavg": clean_fedavg,
        "attacked_fedavg": attacked_fedavg,
        "clean_defense": clean_defense,
        "attacked_defense": attacked_defense,
    }
    round_sets = {
        name: [int(record["round"]) for record in _evaluations(payload)]
        for name, payload in payloads.items()
    }
    if len({tuple(rounds) for rounds in round_sets.values()}) != 1:
        raise ValueError(f"Evaluation rounds differ: {round_sets}")

    finals = {
        name: _final(payload, name) for name, payload in payloads.items()
    }
    final_acc = {
        name: record.get("global_val_acc") for name, record in finals.items()
    }
    class_sets = {
        name: set((record.get("per_class_acc") or {}).keys())
        for name, record in finals.items()
    }
    if len({frozenset(classes) for classes in class_sets.values()}) != 1:
        raise ValueError("Per-class metric sets differ")

    per_class = {}
    for class_name in sorted(next(iter(class_sets.values()))):
        values = {
            name: finals[name]["per_class_acc"][class_name] for name in finals
        }
        per_class[class_name] = {
            **values,
            "clean_defense_penalty": _difference(
                values["clean_defense"], values["clean_fedavg"]
            ),
            "fedavg_attack_damage": _difference(
                values["attacked_fedavg"], values["clean_fedavg"]
            ),
            "defense_attack_damage": _difference(
                values["attacked_defense"], values["clean_defense"]
            ),
            "recovery_vs_attacked_fedavg": _difference(
                values["attacked_defense"], values["attacked_fedavg"]
            ),
        }

    return {
        "experiment_ids": {
            name: payload.get("experiment_id") for name, payload in payloads.items()
        },
        "aggregation": attacked_defense.get("aggregation"),
        "rounds": round_sets["clean_fedavg"],
        "final_accuracy": final_acc,
        "clean_defense_penalty": _difference(
            final_acc["clean_defense"], final_acc["clean_fedavg"]
        ),
        "fedavg_attack_damage": _difference(
            final_acc["attacked_fedavg"], final_acc["clean_fedavg"]
        ),
        "defense_attack_damage": _difference(
            final_acc["attacked_defense"], final_acc["clean_defense"]
        ),
        "recovery_vs_attacked_fedavg": _difference(
            final_acc["attacked_defense"], final_acc["attacked_fedavg"]
        ),
        "per_class_final": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze robust FL aggregation.")
    parser.add_argument("--clean-fedavg", type=Path, required=True)
    parser.add_argument("--attacked-fedavg", type=Path, required=True)
    parser.add_argument("--clean-defense", type=Path, required=True)
    parser.add_argument("--attacked-defense", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = analyze_defense(
            _load(args.clean_fedavg),
            _load(args.attacked_fedavg),
            _load(args.clean_defense),
            _load(args.attacked_defense),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        "[defense-analysis]",
        {
            "aggregation": report["aggregation"],
            "clean_penalty": report["clean_defense_penalty"],
            "attack_damage": report["defense_attack_damage"],
            "recovery": report["recovery_vs_attacked_fedavg"],
            "output": str(args.output_json),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
