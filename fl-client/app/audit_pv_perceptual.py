"""Audit PlantVillage manifests for unresolved perceptual near-duplicates.

This tool is deliberately conservative: it creates candidate pairs but never
merges groups automatically. PlantVillage images have simple backgrounds and
similar leaf shapes, so a perceptual-hash threshold alone is not sufficient
evidence that two files show the same physical leaf.

Example:

    python -m app.audit_pv_perceptual \
        --manifest-dir /protocol/pv27-capped-v1 \
        --output-json /protocol/pv27-capped-v1/perceptual_audit.json
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class BKTree:
    """Small integer BK-tree using Hamming distance."""

    def __init__(self) -> None:
        self.root: Optional[Tuple[int, Dict[int, Tuple]]] = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count() if hasattr(int, "bit_count") else bin(left ^ right).count("1")

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            node_value, children = node
            distance = self.distance(value, node_value)
            if distance == 0:
                return
            child = children.get(distance)
            if child is None:
                children[distance] = (value, {})
                return
            node = child

    def query(self, value: int, radius: int) -> List[Tuple[int, int]]:
        if self.root is None:
            return []
        matches: List[Tuple[int, int]] = []
        pending = [self.root]
        while pending:
            node_value, children = pending.pop()
            distance = self.distance(value, node_value)
            if distance <= radius:
                matches.append((node_value, distance))
            low = distance - radius
            high = distance + radius
            pending.extend(
                child for edge, child in children.items() if low <= edge <= high
            )
        return matches


def difference_hash(image, hash_size: int = 8) -> int:
    from PIL import Image  # type: ignore

    grayscale = image.convert("L").resize(
        (hash_size + 1, hash_size), resample=Image.Resampling.LANCZOS
        if hasattr(Image, "Resampling")
        else Image.LANCZOS
    )
    pixels = list(grayscale.getdata())
    value = 0
    width = hash_size + 1
    for row in range(hash_size):
        offset = row * width
        for column in range(hash_size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def perceptual_hash(image, image_size: int = 32, low_size: int = 8) -> int:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    grayscale = image.convert("L").resize(
        (image_size, image_size),
        resample=Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS,
    )
    pixels = np.asarray(grayscale, dtype=np.float64)
    x = np.arange(image_size, dtype=np.float64)
    u = np.arange(low_size, dtype=np.float64)[:, None]
    basis = np.cos((math.pi / image_size) * (x + 0.5) * u)
    basis[0, :] *= math.sqrt(1.0 / image_size)
    basis[1:, :] *= math.sqrt(2.0 / image_size)
    coefficients = basis @ pixels @ basis.T
    flat = coefficients.flatten()
    median = float(np.median(flat[1:]))
    value = 0
    for coefficient in flat:
        value = (value << 1) | int(float(coefficient) > median)
    return value


def hash_image(path: Path) -> Tuple[int, int]:
    from PIL import Image  # type: ignore

    with Image.open(str(path)) as image:
        image.load()
        return perceptual_hash(image), difference_hash(image)


def _load_records(manifest_dir: Path) -> Tuple[Path, List[Dict], str]:
    records: List[Dict] = []
    raw_roots = set()
    dataset_ids = set()
    for split in ("train", "validation", "test"):
        path = manifest_dir / f"{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_roots.add(str(payload["source"]["raw_root"]))
        dataset_ids.add(str(payload["dataset_id"]))
        for record in payload["records"]:
            if str(record["split"]) != split:
                raise ValueError(f"{path} contains a record assigned to {record['split']}")
            records.append(dict(record))
    if len(raw_roots) != 1 or len(dataset_ids) != 1:
        raise ValueError("Split manifests do not share one source root and dataset ID")
    return Path(next(iter(raw_roots))), records, next(iter(dataset_ids))


def _quantiles(values: Sequence[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": float(ordered[0]),
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "p95": at(0.95),
        "max": float(ordered[-1]),
    }


def audit(
    raw_root: Path,
    records: List[Dict],
    phash_radius: int,
    max_unresolved_candidates: int,
) -> Dict:
    for index, record in enumerate(records):
        source_path = raw_root / str(record["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Manifest image not found: {source_path}")
        phash, dhash = hash_image(source_path)
        record["_index"] = index
        record["_phash"] = phash
        record["_dhash"] = dhash

    known_phash_distances: List[int] = []
    known_dhash_distances: List[int] = []
    records_by_group: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        records_by_group[str(record["group_id"])].append(record)
    known_pair_count = 0
    for group in records_by_group.values():
        if len(group) < 2:
            continue
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                left = group[left_index]
                right = group[right_index]
                known_pair_count += 1
                known_phash_distances.append(
                    BKTree.distance(int(left["_phash"]), int(right["_phash"]))
                )
                known_dhash_distances.append(
                    BKTree.distance(int(left["_dhash"]), int(right["_dhash"]))
                )

    unresolved: List[Dict] = []
    known_candidate_count = 0
    metadata_distinct_candidate_count = 0
    candidate_distance_histogram: Counter = Counter()
    records_by_class: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        records_by_class[str(record["class_name"])].append(record)

    for class_name, class_records in sorted(records_by_class.items()):
        tree = BKTree()
        hash_to_records: Dict[int, List[Dict]] = defaultdict(list)
        for record in sorted(class_records, key=lambda item: str(item["path"])):
            phash = int(record["_phash"])
            for matched_hash, phash_distance in tree.query(phash, phash_radius):
                for previous in hash_to_records[matched_hash]:
                    dhash_distance = BKTree.distance(
                        int(record["_dhash"]), int(previous["_dhash"])
                    )
                    candidate_distance_histogram[
                        f"p{phash_distance:02d}_d{dhash_distance:02d}"
                    ] += 1
                    if record["group_id"] == previous["group_id"]:
                        known_candidate_count += 1
                        continue
                    if record["leaf_id"] is not None and previous["leaf_id"] is not None:
                        # The upstream class-aware metadata identifies these as
                        # different physical leaves. Similarity is recorded in
                        # the histogram but is not an unresolved grouping case.
                        metadata_distinct_candidate_count += 1
                        continue
                    if len(unresolved) < max_unresolved_candidates:
                        unresolved.append(
                            {
                                "class_name": class_name,
                                "left_path": previous["path"],
                                "right_path": record["path"],
                                "left_group_id": previous["group_id"],
                                "right_group_id": record["group_id"],
                                "left_group_source": previous["group_source"],
                                "right_group_source": record["group_source"],
                                "left_split": previous["split"],
                                "right_split": record["split"],
                                "phash_distance": phash_distance,
                                "dhash_distance": dhash_distance,
                                "crosses_split": previous["split"] != record["split"],
                            }
                        )
            if phash not in hash_to_records:
                tree.add(phash)
            hash_to_records[phash].append(record)

    unresolved.sort(
        key=lambda item: (
            int(item["phash_distance"]),
            int(item["dhash_distance"]),
            str(item["class_name"]),
            str(item["left_path"]),
            str(item["right_path"]),
        )
    )
    total_unresolved_seen = (
        sum(candidate_distance_histogram.values())
        - known_candidate_count
        - metadata_distinct_candidate_count
    )
    return {
        "records_hashed": len(records),
        "phash_bits": 64,
        "dhash_bits": 64,
        "candidate_rule": f"same class and pHash Hamming distance <= {phash_radius}",
        "automatic_group_merging": False,
        "known_group_pair_count": known_pair_count,
        "known_group_phash_distance": _quantiles(known_phash_distances),
        "known_group_dhash_distance": _quantiles(known_dhash_distances),
        "known_pairs_retrieved_as_candidates": known_candidate_count,
        "metadata_distinct_candidates": metadata_distinct_candidate_count,
        "unresolved_candidates_found": total_unresolved_seen,
        "unresolved_candidates_written": len(unresolved),
        "unresolved_candidates_truncated": total_unresolved_seen > len(unresolved),
        "unresolved_cross_split_candidates": sum(
            bool(item["crosses_split"]) for item in unresolved
        ),
        "candidate_distance_histogram": dict(sorted(candidate_distance_histogram.items())),
        "unresolved_candidates": unresolved,
        "review_status": "manual_or_stricter_review_required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--phash-radius", type=int, default=8)
    parser.add_argument("--max-unresolved-candidates", type=int, default=50000)
    parser.add_argument(
        "--update-summary",
        action="store_true",
        help="Update summary.json and close the dataset gate only when no unresolved candidates remain.",
    )
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if not manifest_dir.is_dir():
        raise SystemExit(f"--manifest-dir not found: {manifest_dir}")
    if not 0 <= args.phash_radius <= 64:
        raise SystemExit("--phash-radius must be between 0 and 64")
    if args.max_unresolved_candidates <= 0:
        raise SystemExit("--max-unresolved-candidates must be > 0")

    raw_root, records, dataset_id = _load_records(manifest_dir)
    result = audit(
        raw_root=raw_root,
        records=records,
        phash_radius=args.phash_radius,
        max_unresolved_candidates=args.max_unresolved_candidates,
    )
    payload = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "source_manifest_dir": str(manifest_dir),
        "source_raw_root": str(raw_root),
        **result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.update_summary:
        summary_path = manifest_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(summary["dataset_id"]) != dataset_id:
            raise SystemExit("summary.json dataset ID does not match the split manifests")
        unresolved_count = int(result["unresolved_candidates_found"])
        summary["perceptual_audit_status"] = "complete"
        summary["perceptual_audit"] = {
            "path": output_path.name,
            "phash_radius": args.phash_radius,
            "unresolved_candidates": unresolved_count,
            "metadata_distinct_candidates": result["metadata_distinct_candidates"],
        }
        summary["dataset_gate_passed"] = bool(summary["validation"]["passed"]) and unresolved_count == 0
        summary["dataset_gate_note"] = (
            "Structural and perceptual-audit gates passed."
            if summary["dataset_gate_passed"]
            else (
                "Structural checks passed, but unresolved perceptual candidates remain; "
                "the dataset gate is closed."
            )
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(
        "[audit_pv_perceptual]",
        {
            "dataset_id": dataset_id,
            "records": result["records_hashed"],
            "unresolved_candidates": result["unresolved_candidates_found"],
            "cross_split_candidates_written": result["unresolved_cross_split_candidates"],
            "output": str(output_path),
            "automatic_merging": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
