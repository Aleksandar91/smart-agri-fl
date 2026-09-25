"""Evaluate a Flower checkpoint on one locked global dataset manifest.

Unlike round-level Flower validation, this post-run evaluator runs exactly
once per checkpoint and writes per-image predictions required for paired
statistics. A final-test manifest is never used by the training client.

Example:

    python -m app.evaluate_global \
        --data-root /raw/color \
        --manifest /protocol/pv19-capped-primary-v1/test.json \
        --checkpoint /models/global_latest.npz \
        --output-json /models/test_evaluation.json \
        --img-size 128
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from app.field_validate import load_checkpoint_ordered
from app.fl_task import IMAGENET_MEAN, IMAGENET_STD, build_model, set_weights
from app.train_plant_disease import _PathsDataset


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def metrics_from_confusion(confusion: Sequence[Sequence[int]]) -> Dict:
    import numpy as np  # type: ignore

    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be square")
    total = int(matrix.sum())
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    supported = support > 0
    supported_recall = recall[supported]
    supported_f1 = f1[supported]
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "balanced_accuracy": float(supported_recall.mean())
        if supported_recall.size
        else 0.0,
        "macro_f1": float(supported_f1.mean()) if supported_f1.size else 0.0,
        "weighted_f1": float((f1 * support).sum() / support.sum())
        if support.sum()
        else 0.0,
        "worst_class_recall": float(supported_recall.min())
        if supported_recall.size
        else 0.0,
        "p10_class_recall": float(np.quantile(supported_recall, 0.10))
        if supported_recall.size
        else 0.0,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.astype(np.int64).tolist(),
    }


class _EvaluationDataset:
    def __init__(self, samples, transform):
        self.base = _PathsDataset(samples, transform=None)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, label = self.base[index]
        return self.transform(image), label, index


def _environment() -> Dict:
    import PIL  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore
    import torchvision  # type: ignore

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }


def evaluate(
    data_root: Path,
    manifest: Dict,
    checkpoint: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Dict:
    import numpy as np  # type: ignore
    import torch  # type: ignore
    import torch.nn.functional as functional  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
    from torchvision import transforms  # type: ignore

    split = str(manifest.get("split"))
    if split not in ("validation", "test"):
        raise ValueError(
            f"Global evaluation accepts only validation or test manifests, got {split!r}"
        )
    classes = list(manifest["classes"])
    samples = [
        (data_root / str(relative_path), int(label))
        for relative_path, label in manifest["samples"]
    ]
    missing = [str(path) for path, _label in samples if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images are missing, e.g. {missing[:3]}")
    transform = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    dataset = _EvaluationDataset(samples, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    model = build_model(num_classes=len(classes), freeze_backbone=False)
    set_weights(model, load_checkpoint_ordered(checkpoint))
    model.eval()
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    predictions: List[Dict] = []
    total_loss = 0.0
    with torch.no_grad():
        for images, targets, indices in loader:
            logits = model(images)
            total_loss += float(
                functional.cross_entropy(logits, targets, reduction="sum").item()
            )
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            top_values, top_indices = torch.topk(
                probabilities, k=min(3, len(classes)), dim=1
            )
            for batch_index in range(len(targets)):
                target = int(targets[batch_index])
                prediction = int(predicted[batch_index])
                source_index = int(indices[batch_index])
                confusion[target, prediction] += 1
                path = samples[source_index][0]
                predictions.append(
                    {
                        "path": path.relative_to(data_root).as_posix(),
                        "target_index": target,
                        "target_class": classes[target],
                        "predicted_index": prediction,
                        "predicted_class": classes[prediction],
                        "correct": target == prediction,
                        "confidence": float(confidence[batch_index]),
                        "top3": [
                            {
                                "class_index": int(class_index),
                                "class_name": classes[int(class_index)],
                                "probability": float(probability),
                            }
                            for probability, class_index in zip(
                                top_values[batch_index].tolist(),
                                top_indices[batch_index].tolist(),
                            )
                        ],
                    }
                )

    aggregate = metrics_from_confusion(confusion.tolist())
    aggregate["loss"] = total_loss / max(len(dataset), 1)
    per_class = {
        class_name: {
            "precision": aggregate["per_class_precision"][index],
            "recall": aggregate["per_class_recall"][index],
            "f1": aggregate["per_class_f1"][index],
            "support": aggregate["per_class_support"][index],
        }
        for index, class_name in enumerate(classes)
    }
    for key in (
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
        "per_class_support",
    ):
        del aggregate[key]
    return {
        "split": split,
        "num_images": len(dataset),
        "metrics": aggregate,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"--data-root not found: {data_root}")
    if not manifest_path.is_file():
        raise SystemExit(f"--manifest not found: {manifest_path}")
    if not checkpoint.is_file():
        raise SystemExit(f"--checkpoint not found: {checkpoint}")
    if args.img_size < 32 or args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("Invalid img-size, batch-size, or num-workers")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = evaluate(
        data_root=data_root,
        manifest=manifest,
        checkpoint=checkpoint,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    report = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "classes": manifest["classes"],
        "environment": _environment(),
        **evaluation,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "[evaluate_global]",
        {
            "dataset_id": report["dataset_id"],
            "split": report["split"],
            "images": report["num_images"],
            "accuracy": round(report["metrics"]["accuracy"], 6),
            "macro_f1": round(report["metrics"]["macro_f1"], 6),
            "output": str(output_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
