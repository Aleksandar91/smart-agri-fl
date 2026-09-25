"""Build reproducible, group-safe PlantVillage protocol manifests.

The script never copies or modifies source images. It inventories an original
PlantVillage ``raw/color`` tree, normalizes the selected class names, links
images through class-aware ``leaf_id`` metadata and exact SHA-256 duplicates,
and then creates a single global train/validation/test split *before* client
partitioning.

Example:

    python -m app.prepare_pv_protocol \
        --raw-root /raw/PlantVillage-Dataset-master/raw/color \
        --leaf-map /raw/PlantVillage-Dataset-master/leaf-map.json \
        --output-dir /protocol/pv27-capped \
        --variant capped --per-class 300

The generated ``train.json`` is an input pool for ``partition_dataset.py``.
The validation and test manifests must never be partitioned into client
training data.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app.build_pv_subset import CANONICAL_CLASSES, IMG_EXTS


SCHEMA_VERSION = 1
DEFAULT_SUBSET_SEED = 20260824
DEFAULT_SPLIT_SEED = 20260824
SPLIT_NAMES = ("train", "validation", "test")
SOURCE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Tomato___Bacterial_spot": ("Tomato___Bacterial_spot", "Tomato_Bacterial_spot"),
    "Tomato___Early_blight": ("Tomato___Early_blight", "Tomato_Early_blight"),
    "Tomato___healthy": ("Tomato___healthy", "Tomato_healthy"),
    "Tomato___Late_blight": ("Tomato___Late_blight", "Tomato_Late_blight"),
    "Tomato___Leaf_Mold": ("Tomato___Leaf_Mold", "Tomato_Leaf_Mold"),
    "Tomato___Septoria_leaf_spot": (
        "Tomato___Septoria_leaf_spot",
        "Tomato_Septoria_leaf_spot",
    ),
    "Tomato___Spider_mites": (
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
    ),
    "Tomato___Target_Spot": ("Tomato___Target_Spot", "Tomato__Target_Spot"),
    "Tomato___Yellow_Leaf_Curl_Virus": (
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
    ),
    "Tomato___mosaic_virus": (
        "Tomato___Tomato_mosaic_virus",
        "Tomato__Tomato_mosaic_virus",
    ),
}
LEAF_ID_PREFIX_ALIASES: Dict[str, Tuple[str, ...]] = {
    # The official grouping metadata uses the pathological synonym
    # "Frogeye Spot" for the PlantVillage Apple___Black_rot image class.
    "Apple___Black_rot": ("Apple_Frogeye Spot",),
}


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _stable_rng(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _original_stem(path: Path) -> str:
    """Return the stem format used as a key in PlantVillage leaf-map.json."""
    return path.stem.split("___", 1)[-1].lower().strip()


def _class_aware_leaf_id(
    raw_class: str, path: Path, leaf_map: Dict[str, Sequence[str]]
) -> Optional[str]:
    candidates = leaf_map.get(_original_stem(path), ())
    accepted_prefixes = (raw_class,) + LEAF_ID_PREFIX_ALIASES.get(raw_class, ())
    matches = sorted(
        str(value)
        for value in candidates
        if any(str(value).startswith(f"{prefix}:::") for prefix in accepted_prefixes)
    )
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous class-aware leaf IDs for {path}: {matches}. "
            "The source metadata must be resolved before splitting."
        )
    return matches[0]


def _source_classes(selected_classes: Optional[Sequence[str]]) -> Dict[str, Tuple[str, ...]]:
    wanted = set(selected_classes) if selected_classes else None
    aliases: Dict[str, List[str]] = defaultdict(list)
    for raw_name, canonical in CANONICAL_CLASSES.items():
        aliases[canonical].append(raw_name)
    for canonical, raw_names in SOURCE_ALIASES.items():
        aliases[canonical] = list(raw_names)
    known = set(aliases)
    if wanted is not None:
        missing = sorted(wanted - known)
        if missing:
            raise ValueError(f"Unknown canonical classes: {missing}")
    return {
        canonical: tuple(raw_names)
        for canonical, raw_names in sorted(aliases.items())
        if wanted is None or canonical in wanted
    }


def inventory(
    raw_root: Path,
    leaf_map: Dict[str, Sequence[str]],
    selected_classes: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict], Dict]:
    records: List[Dict] = []
    coverage: Dict[str, Dict[str, int]] = {}
    missing_dirs: List[str] = []
    resolved_classes: List[Tuple[str, str]] = []

    for canonical_class, aliases in _source_classes(selected_classes).items():
        existing = [raw_name for raw_name in aliases if (raw_root / raw_name).is_dir()]
        if not existing:
            missing_dirs.append(f"{canonical_class} ({', '.join(aliases)})")
            continue
        if len(existing) > 1:
            raise ValueError(
                f"Multiple source directories map to {canonical_class}: {existing}. "
                "Use a source root with one representation per class."
            )
        resolved_classes.append((existing[0], canonical_class))

    # Fail before hashing any files when the selected source is incomplete.
    if missing_dirs:
        raise FileNotFoundError(f"Selected PlantVillage class directories are missing: {missing_dirs}")

    for raw_class, canonical_class in resolved_classes:
        class_dir = raw_root / raw_class
        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMG_EXTS
        )
        class_mapped = 0
        for path in paths:
            leaf_id = _class_aware_leaf_id(raw_class, path, leaf_map)
            if leaf_id is not None:
                class_mapped += 1
            records.append(
                {
                    "path": path.relative_to(raw_root).as_posix(),
                    "raw_class": raw_class,
                    "class_name": canonical_class,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "leaf_id": leaf_id,
                }
            )
        coverage[canonical_class] = {
            "images": len(paths),
            "leaf_id_mapped": class_mapped,
            "leaf_id_unmapped": len(paths) - class_mapped,
        }

    if not records:
        raise ValueError(f"No PlantVillage images found under {raw_root}")
    return records, coverage


def attach_groups(records: List[Dict]) -> Dict:
    """Connect records by class-aware leaf ID and exact image content."""
    union = UnionFind(len(records))
    by_leaf: Dict[str, List[int]] = defaultdict(list)
    by_hash: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["leaf_id"] is not None:
            by_leaf[str(record["leaf_id"])].append(index)
        by_hash[str(record["sha256"])].append(index)

    for members in list(by_leaf.values()) + list(by_hash.values()):
        anchor = members[0]
        for member in members[1:]:
            union.union(anchor, member)

    components: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(records)):
        components[union.find(index)].append(index)

    conflicting_exact_duplicates: List[Dict] = []
    exact_duplicate_sets = 0
    exact_duplicate_images = 0
    for digest, members in sorted(by_hash.items()):
        if len(members) <= 1:
            continue
        exact_duplicate_sets += 1
        exact_duplicate_images += len(members)
        classes = sorted({str(records[index]["class_name"]) for index in members})
        if len(classes) > 1:
            conflicting_exact_duplicates.append(
                {
                    "sha256": digest,
                    "classes": classes,
                    "paths": sorted(str(records[index]["path"]) for index in members),
                }
            )
    if conflicting_exact_duplicates:
        raise ValueError(
            "Exact duplicate images have conflicting class labels; inspect the audit report data: "
            f"{conflicting_exact_duplicates[:3]}"
        )

    for members in components.values():
        member_paths = sorted(str(records[index]["path"]) for index in members)
        group_digest = hashlib.sha256("\n".join(member_paths).encode("utf-8")).hexdigest()[:20]
        has_leaf_id = any(records[index]["leaf_id"] is not None for index in members)
        has_exact_duplicate = any(len(by_hash[str(records[index]["sha256"])]) > 1 for index in members)
        if has_leaf_id and has_exact_duplicate:
            source = "leaf_id+sha256"
        elif has_leaf_id:
            source = "leaf_id"
        elif has_exact_duplicate:
            source = "sha256"
        else:
            source = "singleton"
        for index in members:
            records[index]["group_id"] = f"pv-{group_digest}"
            records[index]["group_source"] = source

    return {
        "leaf_groups": len(by_leaf),
        "known_leaf_images": sum(1 for record in records if record["leaf_id"] is not None),
        "unknown_leaf_images": sum(1 for record in records if record["leaf_id"] is None),
        "exact_duplicate_sets": exact_duplicate_sets,
        "exact_duplicate_images": exact_duplicate_images,
        "connected_groups": len(components),
        "conflicting_exact_duplicates": conflicting_exact_duplicates,
    }


def summarize_attached_groups(records: Sequence[Dict]) -> Dict:
    hashes: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        hashes[str(record["sha256"])].append(record)
    duplicate_sets = [members for members in hashes.values() if len(members) > 1]
    return {
        "scope": "selected_variant",
        "leaf_groups": len(
            {str(record["leaf_id"]) for record in records if record["leaf_id"] is not None}
        ),
        "known_leaf_images": sum(1 for record in records if record["leaf_id"] is not None),
        "unknown_leaf_images": sum(1 for record in records if record["leaf_id"] is None),
        "exact_duplicate_sets": len(duplicate_sets),
        "exact_duplicate_images": sum(len(members) for members in duplicate_sets),
        "connected_groups": len({str(record["group_id"]) for record in records}),
        "conflicting_exact_duplicates": [],
    }


def select_capped_groups(records: List[Dict], per_class: int, seed: int) -> List[Dict]:
    if per_class <= 0:
        raise ValueError("per_class must be > 0")
    groups_by_class: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        groups_by_class[str(record["class_name"])][str(record["group_id"])].append(record)

    selected: List[Dict] = []
    for class_name in sorted(groups_by_class):
        groups = list(groups_by_class[class_name].values())
        rng = _stable_rng(seed, f"subset:{class_name}")
        rng.shuffle(groups)
        # Prefer larger physical-leaf groups first so the capped benchmark does
        # not systematically reduce multi-view leaves to singleton-heavy data.
        groups.sort(key=len, reverse=True)
        count = 0
        for group in groups:
            if count + len(group) > per_class:
                continue
            selected.extend(group)
            count += len(group)
            if count == per_class:
                break
    return sorted(selected, key=lambda record: str(record["path"]))


def _assign_class_groups(
    groups: List[List[Dict]],
    ratios: Tuple[float, float, float],
    seed: int,
    class_name: str,
) -> Dict[str, List[List[Dict]]]:
    rng = _stable_rng(seed, f"split:{class_name}")
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    total = sum(len(group) for group in groups)
    targets = {
        split: total * ratio for split, ratio in zip(SPLIT_NAMES, ratios)
    }
    assigned: Dict[str, List[List[Dict]]] = {split: [] for split in SPLIT_NAMES}
    counts = {split: 0 for split in SPLIT_NAMES}

    for group in groups:
        candidates = list(SPLIT_NAMES)
        # If possible, reserve enough remaining groups to populate empty splits.
        remaining_after = len(groups) - sum(len(value) for value in assigned.values()) - 1
        empty = [split for split in SPLIT_NAMES if not assigned[split]]
        if empty and remaining_after < len(empty):
            candidates = empty
        chosen = max(
            candidates,
            key=lambda split: (
                (targets[split] - counts[split]) / max(targets[split], 1.0),
                -SPLIT_NAMES.index(split),
            ),
        )
        assigned[chosen].append(group)
        counts[chosen] += len(group)
    return assigned


def assign_splits(
    records: List[Dict],
    ratios: Tuple[float, float, float],
    split_seed: int,
) -> None:
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be > 0")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    by_class_group: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class_group[str(record["class_name"])][str(record["group_id"])].append(record)

    for class_name, grouped in sorted(by_class_group.items()):
        groups = list(grouped.values())
        assigned = _assign_class_groups(groups, ratios, split_seed, class_name)
        for split, split_groups in assigned.items():
            for group in split_groups:
                for record in group:
                    record["split"] = split


def validate(records: Sequence[Dict], classes: Sequence[str]) -> Dict:
    errors: List[str] = []
    group_splits: Dict[str, set] = defaultdict(set)
    hash_splits: Dict[str, set] = defaultdict(set)
    class_split_counts: Dict[str, Dict[str, int]] = {
        class_name: {split: 0 for split in SPLIT_NAMES} for class_name in classes
    }
    class_group_counts: Dict[str, Dict[str, set]] = {
        class_name: {split: set() for split in SPLIT_NAMES} for class_name in classes
    }

    for record in records:
        split = str(record["split"])
        class_name = str(record["class_name"])
        group_splits[str(record["group_id"])].add(split)
        hash_splits[str(record["sha256"])].add(split)
        class_split_counts[class_name][split] += 1
        class_group_counts[class_name][split].add(str(record["group_id"]))

    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    leaking_hashes = sorted(digest for digest, splits in hash_splits.items() if len(splits) > 1)
    if leaking_groups:
        errors.append(f"{len(leaking_groups)} group IDs cross split boundaries")
    if leaking_hashes:
        errors.append(f"{len(leaking_hashes)} SHA-256 values cross split boundaries")
    for class_name, split_counts in class_split_counts.items():
        missing = [split for split, count in split_counts.items() if count == 0]
        if missing:
            errors.append(f"{class_name} has no images in splits: {missing}")

    return {
        "passed": not errors,
        "errors": errors,
        "leaking_group_ids": leaking_groups,
        "leaking_sha256": leaking_hashes,
        "images_per_class_split": class_split_counts,
        "groups_per_class_split": {
            class_name: {
                split: len(groups) for split, groups in split_groups.items()
            }
            for class_name, split_groups in class_group_counts.items()
        },
    }


def _manifest_digest(records: Sequence[Dict]) -> str:
    projection = [
        {
            "path": record["path"],
            "class_name": record["class_name"],
            "sha256": record["sha256"],
            "group_id": record["group_id"],
            "split": record["split"],
        }
        for record in sorted(records, key=lambda item: str(item["path"]))
    ]
    payload = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_outputs(
    output_dir: Path,
    records: Sequence[Dict],
    coverage: Dict,
    grouping_audit: Dict,
    validation: Dict,
    raw_root: Path,
    leaf_map_path: Path,
    variant: str,
    per_class: Optional[int],
    subset_seed: int,
    split_seed: int,
    ratios: Tuple[float, float, float],
    min_leaf_id_coverage: float,
    require_leaf_id: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted({str(record["class_name"]) for record in records})
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    digest = _manifest_digest(records)
    dataset_id = f"pv{len(classes)}-{variant}-{digest[:12]}"

    common = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": digest,
        "source": {
            "raw_root": str(raw_root),
            "leaf_map": str(leaf_map_path),
            "image_variant": "color",
        },
        "selection": {
            "variant": variant,
            "per_class": per_class,
            "subset_seed": subset_seed if variant == "capped" else None,
            "split_seed": split_seed,
            "split_ratios": dict(zip(SPLIT_NAMES, ratios)),
            "min_leaf_id_coverage": min_leaf_id_coverage,
            "require_leaf_id": require_leaf_id,
            "excluded_classes_by_leaf_coverage": sorted(set(coverage) - set(classes)),
        },
        "classes": classes,
        "class_to_idx": class_to_idx,
    }

    for split in SPLIT_NAMES:
        split_records = [
            dict(record)
            for record in sorted(records, key=lambda item: str(item["path"]))
            if record["split"] == split
        ]
        manifest = {
            **common,
            "split": split,
            "samples": [
                [record["path"], class_to_idx[str(record["class_name"])]]
                for record in split_records
            ],
            "records": split_records,
        }
        (output_dir / f"{split}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    summary = {
        **common,
        "total_images": len(records),
        "total_groups": len({str(record["group_id"]) for record in records}),
        "split_images": {
            split: sum(1 for record in records if record["split"] == split)
            for split in SPLIT_NAMES
        },
        "leaf_id_coverage": coverage,
        "grouping_audit": grouping_audit,
        "validation": validation,
        "perceptual_audit_status": "pending",
        "dataset_gate_passed": bool(validation["passed"]) and False,
        "dataset_gate_note": (
            "Structural split checks passed, but the protocol gate remains closed "
            "until the perceptual near-duplicate audit is completed."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _parse_classes(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    classes = [item.strip() for item in value.split(",") if item.strip()]
    return classes or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--leaf-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("capped", "full"), default="capped")
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--subset-seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--min-leaf-id-coverage",
        type=float,
        default=0.0,
        help=(
            "Exclude an entire class when its class-aware leaf-ID coverage is below this "
            "fraction (0.0-1.0). Use 0.99 for the primary leakage-safe benchmark."
        ),
    )
    parser.add_argument(
        "--require-leaf-id",
        action="store_true",
        help="After class filtering, exclude individual images without a class-aware leaf ID.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated canonical classes; default is the existing PV-27 class list.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    leaf_map_path = Path(args.leaf_map).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not raw_root.is_dir():
        raise SystemExit(f"--raw-root not found: {raw_root}")
    if not leaf_map_path.is_file():
        raise SystemExit(f"--leaf-map not found: {leaf_map_path}")
    if not 0.0 <= args.min_leaf_id_coverage <= 1.0:
        raise SystemExit("--min-leaf-id-coverage must be between 0.0 and 1.0")

    leaf_map = json.loads(leaf_map_path.read_text(encoding="utf-8"))
    records, coverage = inventory(raw_root, leaf_map, _parse_classes(args.classes))
    eligible_classes = {
        class_name
        for class_name, values in coverage.items()
        if values["leaf_id_mapped"] / max(values["images"], 1)
        >= args.min_leaf_id_coverage
    }
    records = [
        record
        for record in records
        if record["class_name"] in eligible_classes
        and (not args.require_leaf_id or record["leaf_id"] is not None)
    ]
    if len(eligible_classes) < 2 or not records:
        raise SystemExit(
            "Leaf-ID filtering left fewer than two usable classes. "
            "Lower --min-leaf-id-coverage or inspect the source metadata."
        )
    attach_groups(records)
    if args.variant == "capped":
        records = select_capped_groups(records, args.per_class, args.subset_seed)
    grouping_audit = summarize_attached_groups(records)
    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    assign_splits(records, ratios, args.split_seed)
    classes = sorted({str(record["class_name"]) for record in records})
    validation = validate(records, classes)
    if not validation["passed"]:
        raise SystemExit(f"Dataset split validation failed: {validation['errors']}")
    write_outputs(
        output_dir=output_dir,
        records=records,
        coverage=coverage,
        grouping_audit=grouping_audit,
        validation=validation,
        raw_root=raw_root,
        leaf_map_path=leaf_map_path,
        variant=args.variant,
        per_class=args.per_class if args.variant == "capped" else None,
        subset_seed=args.subset_seed,
        split_seed=args.split_seed,
        ratios=ratios,
        min_leaf_id_coverage=args.min_leaf_id_coverage,
        require_leaf_id=bool(args.require_leaf_id),
    )
    print(
        "[prepare_pv_protocol]",
        {
            "variant": args.variant,
            "images": len(records),
            "groups": len({record["group_id"] for record in records}),
            "output_dir": str(output_dir),
            "structural_validation": "passed",
            "perceptual_audit": "pending",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
