"""Split an ImageFolder-style dataset into per-client partitions for FL experiments.

Instead of copying images, this writes one JSON *manifest* per client
(client_0.json, client_1.json, ...). Every manifest pins the same global class
list and contains the client's samples as (relative_path, label) pairs. Each FL
client then mounts the FULL dataset read-only plus its own manifest - no data
duplication.

Partitioning modes:
  - iid:       every client gets a random, roughly equal share of each class.
  - dirichlet: class proportions per client drawn from Dirichlet(alpha).
               Lower alpha => more heterogeneous (non-IID) clients.
               alpha=0.1 is highly skewed, alpha=100 is close to IID.

Usage:
    python -m app.partition_dataset --data-root /data --output-dir /partitions \
        --num-clients 4 --mode dirichlet --alpha 0.5 --seed 1337
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.train_plant_disease import _list_images_in_dir, _scan_class_folders


def _heterogeneity_summary(
    parts: List[List[Tuple[str, int]]], classes: List[str]
) -> Dict:
    num_clients = len(parts)
    counts = [[0 for _class in classes] for _client in parts]
    for client_index, samples in enumerate(parts):
        for _path, label in samples:
            counts[client_index][label] += 1

    class_metrics = {}
    for label, class_name in enumerate(classes):
        class_counts = [client_counts[label] for client_counts in counts]
        total = sum(class_counts)
        shares = [count / total if total else 0.0 for count in class_counts]
        entropy = -sum(share * math.log(share) for share in shares if share > 0)
        normalized_entropy = (
            entropy / math.log(num_clients) if num_clients > 1 else 0.0
        )
        class_metrics[class_name] = {
            "counts": class_counts,
            "shares": shares,
            "monopoly": max(shares) if shares else 0.0,
            "normalized_entropy": normalized_entropy,
            "effective_clients": math.exp(entropy),
            "clients_with_samples": sum(count > 0 for count in class_counts),
        }

    client_totals = [len(part) for part in parts]
    grand_total = sum(client_totals)
    return {
        "client_sample_counts": client_totals,
        "client_sample_shares": [
            count / grand_total if grand_total else 0.0 for count in client_totals
        ],
        "class_metrics": class_metrics,
        "mean_class_monopoly": sum(
            metrics["monopoly"] for metrics in class_metrics.values()
        )
        / max(len(class_metrics), 1),
        "mean_class_normalized_entropy": sum(
            metrics["normalized_entropy"] for metrics in class_metrics.values()
        )
        / max(len(class_metrics), 1),
    }


def _collect_samples_by_class(data_root: Path, classes: List[str]) -> Dict[int, List[str]]:
    by_class: Dict[int, List[str]] = {}
    for idx, cls in enumerate(classes):
        cls_dir = data_root / cls
        rel_paths = sorted(
            str(p.relative_to(data_root)).replace("\\", "/") for p in _list_images_in_dir(cls_dir)
        )
        by_class[idx] = rel_paths
    return by_class


def _load_grouped_source_manifest(
    manifest_path: Path,
) -> Tuple[Dict, Dict[int, List[List[Tuple[str, int]]]], Dict[str, Dict]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != "train":
        raise ValueError(
            f"Source manifest must be the locked train split, got {manifest.get('split')!r}"
        )
    classes = list(manifest["classes"])
    class_to_idx = {name: index for index, name in enumerate(classes)}
    groups: Dict[int, Dict[str, List[Tuple[str, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    record_by_path: Dict[str, Dict] = {}
    group_class: Dict[str, str] = {}
    for record in manifest.get("records", []):
        path = str(record["path"])
        class_name = str(record["class_name"])
        group_id = str(record["group_id"])
        if class_name not in class_to_idx:
            raise ValueError(f"Record {path} has unknown class {class_name!r}")
        previous_class = group_class.setdefault(group_id, class_name)
        if previous_class != class_name:
            raise ValueError(
                f"Group {group_id} spans classes {previous_class!r} and {class_name!r}"
            )
        label = class_to_idx[class_name]
        groups[label][group_id].append((path, label))
        record_by_path[path] = dict(record)
    if len(record_by_path) != len(manifest.get("samples", [])):
        raise ValueError(
            "Source manifest records/samples differ in size or contain duplicate paths"
        )
    grouped = {
        label: [
            sorted(samples, key=lambda item: item[0])
            for _group_id, samples in sorted(class_groups.items())
        ]
        for label, class_groups in groups.items()
    }
    return manifest, grouped, record_by_path


def _partition_iid(
    by_class: Dict[int, List[str]], num_clients: int, rng: random.Random
) -> List[List[Tuple[str, int]]]:
    parts: List[List[Tuple[str, int]]] = [[] for _ in range(num_clients)]
    for label, paths in by_class.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        for i, p in enumerate(shuffled):
            parts[i % num_clients].append((p, label))
    return parts


def _partition_dirichlet(
    by_class: Dict[int, List[str]],
    num_clients: int,
    alpha: float,
    rng: random.Random,
    min_per_client: int,
) -> List[List[Tuple[str, int]]]:
    import numpy as np  # type: ignore

    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))

    # Retry until every client has at least min_per_client samples
    for _attempt in range(100):
        parts: List[List[Tuple[str, int]]] = [[] for _ in range(num_clients)]
        for label, paths in by_class.items():
            shuffled = paths[:]
            rng.shuffle(shuffled)
            proportions = np_rng.dirichlet([alpha] * num_clients)
            # cumulative cut points over this class's samples
            cuts = (np.cumsum(proportions) * len(shuffled)).astype(int)[:-1]
            chunks = np.split(np.array(shuffled, dtype=object), cuts)
            for client_idx, chunk in enumerate(chunks):
                parts[client_idx].extend((str(p), label) for p in chunk)
        if all(len(p) >= min_per_client for p in parts):
            return parts
    raise SystemExit(
        f"Could not satisfy min {min_per_client} samples/client after 100 attempts. "
        f"Increase --alpha, reduce --num-clients, or lower --min-per-client."
    )


def _partition_grouped_iid(
    by_class_groups: Dict[int, List[List[Tuple[str, int]]]],
    num_clients: int,
    rng: random.Random,
) -> List[List[Tuple[str, int]]]:
    parts: List[List[Tuple[str, int]]] = [[] for _ in range(num_clients)]
    per_class_counts: List[Dict[int, int]] = [defaultdict(int) for _ in range(num_clients)]
    for label, source_groups in by_class_groups.items():
        groups = source_groups[:]
        rng.shuffle(groups)
        for group in groups:
            client_index = min(
                range(num_clients),
                key=lambda index: (per_class_counts[index][label], index),
            )
            parts[client_index].extend(group)
            per_class_counts[client_index][label] += len(group)
    return parts


def _partition_grouped_dirichlet(
    by_class_groups: Dict[int, List[List[Tuple[str, int]]]],
    num_clients: int,
    alpha: float,
    rng: random.Random,
    min_per_client: int,
) -> List[List[Tuple[str, int]]]:
    import numpy as np  # type: ignore

    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    for _attempt in range(100):
        parts: List[List[Tuple[str, int]]] = [[] for _ in range(num_clients)]
        for label, source_groups in by_class_groups.items():
            groups = source_groups[:]
            rng.shuffle(groups)
            proportions = np_rng.dirichlet([alpha] * num_clients)
            total_images = sum(len(group) for group in groups)
            targets = [float(value) * total_images for value in proportions]
            assigned = [0] * num_clients
            for group in groups:
                client_index = max(
                    range(num_clients),
                    key=lambda index: (
                        targets[index] - assigned[index],
                        -assigned[index],
                        -index,
                    ),
                )
                parts[client_index].extend(group)
                assigned[client_index] += len(group)
        if all(len(part) >= min_per_client for part in parts):
            return parts
    raise SystemExit(
        f"Could not satisfy min {min_per_client} group-safe samples/client after 100 attempts. "
        f"Increase --alpha, reduce --num-clients, or lower --min-per-client."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create per-client FL partition manifests.")
    parser.add_argument("--data-root", type=str, default=None, help="Legacy dataset root with class subfolders.")
    parser.add_argument(
        "--source-manifest",
        type=str,
        default=None,
        help="Locked train.json from prepare_pv_protocol.py (recommended).",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write client_*.json.")
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["iid", "dirichlet"], default="dirichlet")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration (lower = more non-IID).")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-per-client", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    if bool(args.data_root) == bool(args.source_manifest):
        raise SystemExit("Provide exactly one of --data-root or --source-manifest")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    source_manifest: Optional[Dict] = None
    record_by_path: Dict[str, Dict] = {}
    if args.source_manifest:
        source_manifest_path = Path(args.source_manifest).expanduser().resolve()
        if not source_manifest_path.is_file():
            raise SystemExit(f"--source-manifest not found: {source_manifest_path}")
        source_manifest, by_class_groups, record_by_path = _load_grouped_source_manifest(
            source_manifest_path
        )
        classes = list(source_manifest["classes"])
        data_root_value = str(source_manifest["source"]["raw_root"])
        total = sum(
            len(group)
            for groups in by_class_groups.values()
            for group in groups
        )
        if args.mode == "iid":
            parts = _partition_grouped_iid(by_class_groups, args.num_clients, rng)
        else:
            parts = _partition_grouped_dirichlet(
                by_class_groups,
                args.num_clients,
                args.alpha,
                rng,
                args.min_per_client,
            )
    else:
        data_root = Path(args.data_root).expanduser().resolve()
        if not data_root.is_dir():
            raise SystemExit(f"--data-root not found: {data_root}")
        classes = _scan_class_folders(data_root)
        if len(classes) < 2:
            raise SystemExit(
                f"Need at least 2 class folders under {data_root}, found {len(classes)}"
            )
        by_class = _collect_samples_by_class(data_root, classes)
        total = sum(len(value) for value in by_class.values())
        data_root_value = str(data_root)
        if args.mode == "iid":
            parts = _partition_iid(by_class, args.num_clients, rng)
        else:
            parts = _partition_dirichlet(
                by_class,
                args.num_clients,
                args.alpha,
                rng,
                args.min_per_client,
            )

    summary = {
        "data_root": data_root_value,
        "source_manifest": str(Path(args.source_manifest).expanduser().resolve())
        if args.source_manifest
        else None,
        "source_dataset_id": source_manifest.get("dataset_id")
        if source_manifest
        else None,
        "source_dataset_manifest_sha256": source_manifest.get(
            "dataset_manifest_sha256"
        )
        if source_manifest
        else None,
        "source_split": source_manifest.get("split") if source_manifest else None,
        "group_safe": bool(source_manifest),
        "mode": args.mode,
        "alpha": args.alpha if args.mode == "dirichlet" else None,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "classes": classes,
        "total_samples": total,
        "clients": [],
        "heterogeneity": _heterogeneity_summary(parts, classes),
    }

    seen_groups: Dict[str, int] = {}
    for i, samples in enumerate(parts):
        per_class = defaultdict(int)
        for _p, label in samples:
            per_class[classes[label]] += 1
        manifest = {
            "client_id": i,
            "mode": args.mode,
            "alpha": args.alpha if args.mode == "dirichlet" else None,
            "seed": args.seed,
            "classes": classes,  # GLOBAL class list, identical for all clients
            "samples": samples,
            "source_dataset_id": source_manifest.get("dataset_id")
            if source_manifest
            else None,
            "source_dataset_manifest_sha256": source_manifest.get(
                "dataset_manifest_sha256"
            )
            if source_manifest
            else None,
            "source_split": source_manifest.get("split") if source_manifest else None,
            "group_safe": bool(source_manifest),
        }
        if source_manifest:
            manifest["records"] = [record_by_path[path] for path, _label in samples]
            for record in manifest["records"]:
                group_id = str(record["group_id"])
                previous_client = seen_groups.setdefault(group_id, i)
                if previous_client != i:
                    raise ValueError(
                        f"Group {group_id} crosses clients {previous_client} and {i}"
                    )
        path = out_dir / f"client_{i}.json"
        path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        summary["clients"].append(
            {
                "client_id": i,
                "num_samples": len(samples),
                "num_groups": len(
                    {str(record["group_id"]) for record in manifest.get("records", [])}
                )
                if source_manifest
                else None,
                "per_class": dict(per_class),
            }
        )
        print(f"[partition] client_{i}: {len(samples)} samples, per_class={dict(per_class)}")

    if source_manifest:
        expected_groups = {
            str(record["group_id"]) for record in source_manifest["records"]
        }
        if set(seen_groups) != expected_groups:
            raise ValueError("Not all source groups were assigned exactly once")
        summary["total_groups"] = len(expected_groups)
        summary["assigned_groups"] = len(seen_groups)
    (out_dir / "partitions_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[partition] wrote {args.num_clients} manifests + summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
