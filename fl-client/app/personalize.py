"""Personalize the FL global model: fine-tune on real farm photos, then
measure whether that closes the "healthy detection" domain gap found by
``field_validate.py``.

Only species with a *complete* PV mapping (both a healthy AND a disease
class, see ``field_validate.FARM_TO_PV``) and sufficient independent capture
sessions are used, because fine-tuning needs both classes and a leakage-free
test session to learn/evaluate a real decision boundary. Incomplete or
under-sampled species remain evaluation-only.

Pseudo-labeling (we have no lab-confirmed disease diagnosis for farm photos):

  - "healthy" farm photo  -> pseudo-label = that species' single PV healthy
    class (unambiguous).
  - "suspect" farm photo  -> pseudo-label = whichever of that species' PV
    disease classes the *current* (pre-finetune) model scores highest
    (self-training / restricted-softmax argmax). We don't know the exact
    disease subtype, so we let the model's own visual features pick the
    closest-matching one instead of forcing every "suspect" image onto the
    same disease class.

This is standard self-training-style domain adaptation: the pseudo-labels
are only used for the TRAIN split. By default, the latest capture date is
held out as TEST and earlier dates are used for training, preventing
near-identical burst photos of the same leaf from leaking across the split.
Evaluation uses the same coarse, ground-truth-based healthy_match/
disease_match/other_class bucket metric as ``field_validate.py`` - never
pseudo-labels.

Usage:

    python -m app.personalize \\
        --checkpoint /models/global_latest.npz \\
        --partition-manifest /partitions/client_0.json \\
        --data-root /farm \\
        --img-size 128 \\
        --test-frac 0.3 \\
        --epochs 8 \\
        --output-dir /out
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from app.field_validate import (
    FARM_TO_PV,
    classify_bucket,
    iter_farm_images,
    load_checkpoint_ordered,
    load_image_tensor,
    predict_topk,
)
from app.fl_task import build_model, get_weights, set_weights

# Species usable for fine-tuning: need BOTH a healthy and a disease PV class.
FULLY_MAPPED_SPECIES = sorted(
    sp for sp, m in FARM_TO_PV.items() if m["healthy"] and m["disease"]
)


def _capture_date(path: Path) -> str:
    """Extract YYYYMMDD from IMG_YYYYMMDD_HHMMSS filenames."""
    match = re.search(r"IMG_(\d{8})_\d{6}", path.stem)
    return match.group(1) if match else "unknown"


def _stratified_split(
    paths: List[Path],
    test_frac: float,
    seed: int,
    split_mode: str,
) -> Tuple[List[Path], List[Path]]:
    if split_mode == "date":
        by_date: Dict[str, List[Path]] = defaultdict(list)
        for path in sorted(paths):
            by_date[_capture_date(path)].append(path)
        dates = sorted(by_date)
        if len(dates) < 2:
            # No independent earlier/later capture sessions: evaluation-only.
            # Returning an empty train set prevents near-duplicate leakage.
            return [], sorted(paths)
        test_date = dates[-1]
        train = [path for date in dates[:-1] for path in by_date[date]]
        return train, sorted(by_date[test_date])

    rnd = random.Random(seed)
    shuffled = sorted(paths)
    rnd.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_frac))
    return shuffled[n_test:], shuffled[:n_test]  # train, test


def build_splits(
    data_root: Path,
    species_list: List[str],
    test_frac: float,
    seed: int,
    split_mode: str = "date",
):
    """Returns {(species, label): {"train": [...], "test": [...]}}, split per (species, label)
    so both healthy and suspect are represented in train and test.

    ``date`` (recommended) holds out the latest YYYYMMDD capture session and
    trains on earlier dates. This prevents near-identical burst photos of the
    same leaf from leaking across train/test. ``random`` preserves the old
    behavior only for backward comparison.
    """
    by_key: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for species, label, path in iter_farm_images(data_root):
        if species in species_list:
            by_key[(species, label)].append(path)

    splits = {}
    for key, paths in by_key.items():
        train, test = _stratified_split(paths, test_frac, seed, split_mode)
        splits[key] = {"train": train, "test": test}
    return splits


def eligible_finetune_species(
    splits: dict,
    candidates: List[str],
    min_train_per_label: int,
    min_test_per_label: int,
) -> Tuple[List[str], Dict[str, str]]:
    """Require independent train/test evidence for BOTH healthy and suspect."""
    eligible: List[str] = []
    excluded: Dict[str, str] = {}
    for species in candidates:
        problems = []
        for label in ("healthy", "suspect"):
            parts = splits.get((species, label))
            if parts is None:
                problems.append(f"missing {label} folder")
                continue
            if len(parts["train"]) < min_train_per_label:
                problems.append(
                    f"{label} train={len(parts['train'])} < {min_train_per_label}"
                )
            if len(parts["test"]) < min_test_per_label:
                problems.append(
                    f"{label} test={len(parts['test'])} < {min_test_per_label}"
                )
        if problems:
            excluded[species] = "; ".join(problems)
        else:
            eligible.append(species)
    return eligible, excluded


def make_pseudo_labeled_samples(model, classes: List[str], splits: dict, img_size: int):
    """Train split only -> list of (img_path, class_index) using the pseudo-labeling
    rules described in the module docstring."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples: List[Tuple[Path, int]] = []
    for (species, label), parts in splits.items():
        mapping = FARM_TO_PV[species]
        for img_path in parts["train"]:
            if label == "healthy":
                target = mapping["healthy"][0]
            else:
                candidates = mapping["disease"]
                if len(candidates) == 1:
                    target = candidates[0]
                else:
                    top = predict_topk(model, img_path, classes, img_size, k=len(classes))
                    ranked = [c for c, _ in top if c in candidates]
                    target = ranked[0] if ranked else candidates[0]
            samples.append((img_path, class_to_idx[target]))
    return samples


def sample_pv_anchors(
    pv_data_root: Path,
    classes: List[str],
    covered_counts: Dict[str, int],
    per_class: int,
    seed: int,
) -> List[Tuple[Path, int]]:
    """Anchor ("replay") examples from the original PlantVillage folders for any
    class that got ZERO farm training images this round (e.g. Blueberry___healthy,
    Squash___Powdery_mildew when only Apple/Tomato are fine-tuned).

    Without this, cross-entropy fine-tuning on a subset of classes pushes those
    classes' logits down for every image (softmax is a zero-sum game across all
    8 classes) - a textbook case of catastrophic forgetting. Mixing in a handful
    of original samples for the untouched classes keeps their decision boundary
    anchored while the exposed classes adapt to the farm domain.
    """
    rnd = random.Random(seed)
    anchors: List[Tuple[Path, int]] = []
    for idx, cls in enumerate(classes):
        if covered_counts.get(cls, 0) > 0:
            continue
        class_dir = pv_data_root / cls
        if not class_dir.is_dir():
            continue
        imgs = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.JPG"))
        if not imgs:
            continue
        chosen = rnd.sample(imgs, min(per_class, len(imgs)))
        anchors.extend((p, idx) for p in chosen)
    return anchors


def fine_tune(model, samples: List[Tuple[Path, int]], img_size: int, epochs: int, lr: float):
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torchvision import transforms  # type: ignore
    from PIL import Image  # type: ignore

    from app.fl_task import IMAGENET_MEAN, IMAGENET_STD

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    class _FarmDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, y = self.items[i]
            img = Image.open(path).convert("RGB")
            return train_tf(img), y

    loader = torch.utils.data.DataLoader(
        _FarmDataset(samples), batch_size=8, shuffle=True, num_workers=0
    )

    device = torch.device("cpu")
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        running, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            bs = int(yb.shape[0])
            running += float(loss.item()) * bs
            seen += bs
        print(f"  epoch {epoch + 1}/{epochs}  train_loss={running / max(1, seen):.4f}")
    model.eval()


def evaluate_on_test(model, classes: List[str], splits: dict, img_size: int) -> Dict:
    """Same coarse bucket metric as field_validate.py, but only on the held-out
    test images (real farm labels, never seen in training - not pseudo-labels)."""
    out = {}
    for (species, label), parts in sorted(splits.items()):
        counts = defaultdict(int)
        for img_path in parts["test"]:
            top1_class, _ = predict_topk(model, img_path, classes, img_size, k=1)[0]
            bucket = classify_bucket(species, top1_class)
            counts[bucket] += 1
        total = sum(counts.values())
        expected = "healthy_match" if label == "healthy" else "disease_match"
        out[f"{species}/{label}"] = {"n": total, **dict(counts)}
        expected_pv_classes = FARM_TO_PV[species]["healthy" if label == "healthy" else "disease"]
        if not expected_pv_classes:
            out[f"{species}/{label}"]["note"] = "no matching PV class for this species (see field_validate.FARM_TO_PV)"
        else:
            out[f"{species}/{label}"]["expected_match_rate"] = round(counts.get(expected, 0) / max(1, total), 3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--partition-manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument(
        "--split-mode",
        choices=["date", "random"],
        default="date",
        help="date (recommended): hold out latest capture date; random: legacy per-image split",
    )
    p.add_argument("--min-train-per-label", type=int, default=5)
    p.add_argument("--min-test-per-label", type=int, default=5)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument(
        "--freeze-backbone",
        type=int,
        default=1,
        help="1 = only fine-tune the classifier head (recommended for ~150 images), 0 = full fine-tune",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--pv-anchor-root",
        default=None,
        help="Path to the original PlantVillage class folders (e.g. /data from fl-data-pv). "
        "If set, adds --anchor-per-class replay images for any class with zero farm "
        "training images, to prevent catastrophic forgetting of those classes.",
    )
    p.add_argument("--anchor-per-class", type=int, default=20)
    p.add_argument(
        "--eval-species",
        default=None,
        help="Comma-separated species to include in the BEFORE/AFTER eval table "
        "(default: all species with any PV mapping, so you can see side-effects "
        "on species that were NOT fine-tuned)",
    )
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    manifest = json.loads(Path(args.partition_manifest).read_text(encoding="utf-8"))
    classes = list(manifest["classes"])
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_splits = build_splits(
        data_root,
        FULLY_MAPPED_SPECIES,
        args.test_frac,
        args.seed,
        args.split_mode,
    )
    fine_tuned_species, excluded_species = eligible_finetune_species(
        candidate_splits,
        FULLY_MAPPED_SPECIES,
        args.min_train_per_label,
        args.min_test_per_label,
    )
    ft_splits = {
        key: parts
        for key, parts in candidate_splits.items()
        if key[0] in fine_tuned_species
    }
    print(f"Split mode: {args.split_mode}")
    print(f"Fine-tune species: {fine_tuned_species}")
    if excluded_species:
        print(f"Excluded from fine-tuning: {excluded_species}")
    for (species, label), parts in sorted(ft_splits.items()):
        train_dates = sorted({_capture_date(p) for p in parts["train"]})
        test_dates = sorted({_capture_date(p) for p in parts["test"]})
        print(
            f"  {species}/{label}: train={len(parts['train'])} dates={train_dates} "
            f"test={len(parts['test'])} dates={test_dates}"
        )

    eval_species = (
        [s.strip() for s in args.eval_species.split(",")]
        if args.eval_species
        else sorted(FARM_TO_PV.keys())
    )
    # Species outside FULLY_MAPPED_SPECIES get an empty train split (eval-only, no fine-tuning).
    eval_only_species = [s for s in eval_species if s not in fine_tuned_species]
    # Keep excluded fully-mapped species (e.g. Pepper with only one suspect
    # session) in the report as evaluation-only, but never train on them.
    all_splits = dict(candidate_splits)
    if eval_only_species:
        eval_only_splits = build_splits(
            data_root,
            [s for s in eval_only_species if s not in FULLY_MAPPED_SPECIES],
            args.test_frac,
            args.seed,
            args.split_mode,
        )
        all_splits.update(eval_only_splits)

    print("\nLoading baseline (pre-finetune) checkpoint for BEFORE eval + pseudo-labeling...")
    base_model = build_model(num_classes=len(classes), freeze_backbone=bool(args.freeze_backbone))
    set_weights(base_model, load_checkpoint_ordered(Path(args.checkpoint)))
    base_model.eval()

    before = evaluate_on_test(base_model, classes, all_splits, args.img_size)

    print("\nPseudo-labeling train split using the baseline model...")
    train_samples = make_pseudo_labeled_samples(base_model, classes, ft_splits, args.img_size)
    n_farm_train_images = len(train_samples)
    print(f"  {n_farm_train_images} pseudo-labeled training images")

    label_counts = defaultdict(int)
    for _, idx in train_samples:
        label_counts[classes[idx]] += 1
    print(f"  pseudo-label distribution: {dict(label_counts)}")

    anchor_samples: List[Tuple[Path, int]] = []
    if args.pv_anchor_root:
        anchor_samples = sample_pv_anchors(
            Path(args.pv_anchor_root), classes, label_counts, args.anchor_per_class, args.seed
        )
        anchored_classes = sorted({classes[idx] for _, idx in anchor_samples})
        print(f"  + {len(anchor_samples)} PV anchor images for untouched classes: {anchored_classes}")
        train_samples = train_samples + anchor_samples
        for _, idx in anchor_samples:
            label_counts[classes[idx]] += 1

    print(f"\nFine-tuning ({'head only' if args.freeze_backbone else 'full network'}, {args.epochs} epochs)...")
    fine_tune(base_model, train_samples, args.img_size, args.epochs, args.lr)

    print("\nEvaluating AFTER fine-tuning (same held-out test images)...")
    after = evaluate_on_test(base_model, classes, all_splits, args.img_size)

    comparison = {}
    for key in sorted(set(before) | set(after)):
        comparison[key] = {"before": before.get(key), "after": after.get(key)}

    report = {
        "checkpoint": str(args.checkpoint),
        "img_size": args.img_size,
        "split_mode": args.split_mode,
        "fine_tuned_species": fine_tuned_species,
        "excluded_species": excluded_species,
        "eval_only_species": eval_only_species,
        "epochs": args.epochs,
        "lr": args.lr,
        "freeze_backbone": bool(args.freeze_backbone),
        "n_train_images": len(train_samples),
        "n_farm_train_images": n_farm_train_images,
        "n_anchor_images": len(anchor_samples),
        "pseudo_label_distribution": dict(label_counts),
        "comparison": comparison,
    }

    print("\n=== BEFORE vs AFTER (expected_match_rate) ===")
    for key, v in comparison.items():
        b = (v["before"] or {}).get("expected_match_rate")
        a = (v["after"] or {}).get("expected_match_rate")
        print(f"  {key}: before={b}  after={a}")

    (out_dir / "personalize_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    import numpy as np  # type: ignore

    np.savez(str(out_dir / "personalized_model.npz"), *get_weights(base_model))

    split_manifest = {
        f"{sp}/{label}": {"train": [str(p) for p in parts["train"]], "test": [str(p) for p in parts["test"]]}
        for (sp, label), parts in all_splits.items()
    }
    (out_dir / "train_test_split.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {out_dir / 'personalize_report.json'}")
    print(f"Wrote {out_dir / 'personalized_model.npz'}")
    print(f"Wrote {out_dir / 'train_test_split.json'}")


if __name__ == "__main__":
    main()
