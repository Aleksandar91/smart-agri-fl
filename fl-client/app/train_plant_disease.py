import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class TrainConfig:
    arch: str
    img_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    val_split: float
    freeze_backbone: bool
    num_workers: int


def _list_images_in_dir(d: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _infer_dataset_layout(data_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Returns (train_dir, val_dir) if split exists, otherwise (None, None).
    """
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if train_dir.is_dir() and val_dir.is_dir():
        return train_dir, val_dir
    return None, None


def _load_imagefolder_dataset(
    root: Path,
    transform,
):
    from torchvision.datasets import ImageFolder  # type: ignore

    return ImageFolder(str(root), transform=transform)


def _build_mobilenet_v3_small(num_classes: int):
    import torch.nn as nn  # type: ignore
    from torchvision import models  # type: ignore

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model, weights


class _TransformOnlyDataset:
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        # base returns (PIL image, label) if it has no transform; or tensor if it does
        img, label = self.base[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class _PathsDataset:
    def __init__(self, samples: Sequence[Tuple[Path, int]], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image  # type: ignore

        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def _scan_class_folders(root: Path) -> List[str]:
    classes = []
    for child in root.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            classes.append(child.name)
    classes.sort()
    return classes


def _build_samples_from_class_folders(root: Path, class_to_idx: Dict[str, int]) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for cls_name, idx in class_to_idx.items():
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            continue
        for p in _list_images_in_dir(cls_dir):
            samples.append((p, idx))
    return samples


def _split_indices(n: int, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    val_n = int(n * val_split)
    val_idxs = idxs[:val_n]
    train_idxs = idxs[val_n:]
    return train_idxs, val_idxs


def _subset_dataset(ds, indices: Sequence[int]):
    from torch.utils.data import Subset  # type: ignore

    return Subset(ds, list(indices))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a lightweight plant disease classifier (CPU-friendly) using torchvision transfer learning."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root. Either contains train/ and val/ ImageFolder splits, or class subfolders for auto-split.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write checkpoints/labels/metrics.",
    )
    parser.add_argument(
        "--local-additions",
        type=str,
        default="",
        help="Optional ImageFolder-like directory with extra labeled images to include in training only.",
    )
    parser.add_argument("--arch", type=str, default=os.getenv("PLANT_ARCH", "mobilenet_v3_small"))
    parser.add_argument("--img-size", type=int, default=_env_int("PLANT_IMG_SIZE", 224))
    parser.add_argument("--batch-size", type=int, default=_env_int("PLANT_BATCH_SIZE", 16))
    parser.add_argument("--epochs", type=int, default=_env_int("PLANT_EPOCHS", 5))
    parser.add_argument("--lr", type=float, default=_env_float("PLANT_LR", 1e-3))
    parser.add_argument("--weight-decay", type=float, default=_env_float("PLANT_WEIGHT_DECAY", 1e-4))
    parser.add_argument("--seed", type=int, default=_env_int("PLANT_SEED", 1337))
    parser.add_argument("--val-split", type=float, default=_env_float("PLANT_VAL_SPLIT", 0.15))
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=os.getenv("PLANT_FREEZE_BACKBONE", "true").strip().lower() in {"1", "true", "t", "yes", "y", "on"},
        help="Freeze feature extractor for faster CPU training (default true via env PLANT_FREEZE_BACKBONE).",
    )
    parser.add_argument("--num-workers", type=int, default=_env_int("PLANT_NUM_WORKERS", 2))

    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    local_additions = Path(args.local_additions).expanduser().resolve() if args.local_additions else None

    if not data_root.is_dir():
        raise SystemExit(f"--data-root not found: {data_root}")
    if local_additions is not None and not local_additions.is_dir():
        raise SystemExit(f"--local-additions not found: {local_additions}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        arch=str(args.arch),
        img_size=max(32, int(args.img_size)),
        batch_size=max(1, int(args.batch_size)),
        epochs=max(1, int(args.epochs)),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        val_split=float(args.val_split),
        freeze_backbone=bool(args.freeze_backbone),
        num_workers=max(0, int(args.num_workers)),
    )

    _seed_everything(cfg.seed)

    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import ConcatDataset, DataLoader  # type: ignore
    from torchvision import transforms  # type: ignore

    device = torch.device("cpu")

    if cfg.arch != "mobilenet_v3_small":
        raise SystemExit(f"Unsupported --arch {cfg.arch!r}. Supported: mobilenet_v3_small")

    model, _ = _build_mobilenet_v3_small(num_classes=2)  # placeholder until classes known

    # ImageNet normalization (used by torchvision pretrained models; stable across versions)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(cfg.img_size + 32),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_dir, val_dir = _infer_dataset_layout(data_root)

    if train_dir is not None and val_dir is not None:
        # Pre-split ImageFolder
        train_ds = _load_imagefolder_dataset(train_dir, transform=train_tf)
        val_ds = _load_imagefolder_dataset(val_dir, transform=val_tf)
        class_to_idx = dict(train_ds.class_to_idx)
        classes = list(train_ds.classes)
    else:
        # Auto-split from class folders in data_root
        classes = _scan_class_folders(data_root)
        if not classes:
            raise SystemExit(
                "Dataset layout not recognized. Expected data_root/train & data_root/val or class subfolders."
            )
        class_to_idx = {c: i for i, c in enumerate(classes)}
        all_samples = _build_samples_from_class_folders(data_root, class_to_idx)
        if len(all_samples) < 10:
            raise SystemExit(f"Not enough images found under {data_root} (found {len(all_samples)}).")

        base_ds = _PathsDataset(all_samples, transform=None)  # no transform yet
        train_idxs, val_idxs = _split_indices(len(base_ds), cfg.val_split, cfg.seed)
        train_ds = _TransformOnlyDataset(_subset_dataset(base_ds, train_idxs), transform=train_tf)
        val_ds = _TransformOnlyDataset(_subset_dataset(base_ds, val_idxs), transform=val_tf)

    num_classes = len(classes)
    if num_classes < 2:
        raise SystemExit(f"Need at least 2 classes; found {num_classes} in {data_root}")

    # Rebuild model with correct classifier head
    model, _ = _build_mobilenet_v3_small(num_classes=num_classes)

    if cfg.freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    # Optional local additions (training-only)
    if local_additions is not None:
        add_classes = _scan_class_folders(local_additions)
        unknown = [c for c in add_classes if c not in class_to_idx]
        if unknown:
            raise SystemExit(
                "local-additions contains classes not in base dataset: "
                + ", ".join(unknown)
                + ". Add them to base dataset first or rename folders."
            )
        add_samples = _build_samples_from_class_folders(local_additions, class_to_idx)
        if add_samples:
            add_ds = _PathsDataset(add_samples, transform=train_tf)
            train_ds = ConcatDataset([train_ds, add_ds])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    from tqdm import tqdm  # type: ignore

    best_acc = -1.0
    history: List[Dict] = []

    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps({"classes": classes, "class_to_idx": class_to_idx}, indent=2), encoding="utf-8")

    cfg_path = out_dir / "train_config.json"
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    def evaluate() -> Tuple[float, float]:
        model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss_sum += float(loss.item()) * int(yb.shape[0])
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == yb).sum().item())
                total += int(yb.shape[0])
        avg_loss = loss_sum / max(1, total)
        acc = correct / max(1, total)
        return avg_loss, acc

    print(
        "[train-plant-disease] starting",
        {
            "data_root": str(data_root),
            "local_additions": str(local_additions) if local_additions else None,
            "output_dir": str(out_dir),
            "classes": len(classes),
            "arch": cfg.arch,
            "img_size": cfg.img_size,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "freeze_backbone": cfg.freeze_backbone,
        },
    )

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", unit="batch")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = int(yb.shape[0])
            seen += bs
            running_loss += float(loss.item()) * bs
            pbar.set_postfix({"loss": running_loss / max(1, seen)})

        train_loss = running_loss / max(1, seen)
        val_loss, val_acc = evaluate()
        took_s = time.time() - t0

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "took_s": took_s,
        }
        history.append(rec)

        (out_dir / "metrics.json").write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")

        ckpt = {
            "arch": cfg.arch,
            "img_size": cfg.img_size,
            "classes": classes,
            "class_to_idx": class_to_idx,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "train_config": asdict(cfg),
        }
        torch.save(ckpt, out_dir / "last.pt")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt["best_acc"] = best_acc
            torch.save(ckpt, out_dir / "best.pt")

        print("[train-plant-disease] epoch_done", rec, {"best_acc": best_acc})

    print("[train-plant-disease] done", {"best_acc": best_acc, "output_dir": str(out_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

