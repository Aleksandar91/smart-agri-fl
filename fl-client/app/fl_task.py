"""Shared ML logic for the Flower FL client.

Wraps the existing local-training building blocks from ``train_plant_disease.py``
(model factory, dataset helpers, transforms) into small functions that the
Flower client (``fl_client.py``) can call each round:

  - build_model / get_weights / set_weights   (model <-> list of ndarrays)
  - load_data                                 (full ImageFolder-style root OR a partition manifest)
  - train_one_round / evaluate_model          (local epochs, val metrics)

A "partition manifest" is a JSON file produced by ``partition_dataset.py``.
It pins a *global* class list (shared by all clients) and lists this client's
image paths relative to the data root, so every client's model head has the
same shape even when a client is missing some classes locally (non-IID).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app.train_plant_disease import (
    _build_mobilenet_v3_small,
    _build_samples_from_class_folders,
    _PathsDataset,
    _scan_class_folders,
    _split_indices,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _TransformDataset:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]
        return self.transform(image), label


class _LabelFlipDataset:
    """Training-only label remapping with explicit federated-round gating."""

    def __init__(
        self,
        dataset,
        poisoned_positions,
        source_label: int,
        target_label: int,
        source_count: int,
        start_round: int,
        end_round: Optional[int],
    ):
        self.dataset = dataset
        self.poisoned_positions = frozenset(poisoned_positions)
        self.source_label = int(source_label)
        self.target_label = int(target_label)
        self.source_count = int(source_count)
        self.start_round = int(start_round)
        self.end_round = int(end_round) if end_round is not None else None
        self.current_round = 0

    def __len__(self):
        return len(self.dataset)

    def set_round(self, server_round: int) -> None:
        self.current_round = int(server_round)

    def is_active(self) -> bool:
        if self.current_round < self.start_round:
            return False
        return self.end_round is None or self.current_round <= self.end_round

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.is_active() and index in self.poisoned_positions:
            if int(label) != self.source_label:
                raise RuntimeError("Selected label-flip position no longer has the source label")
            label = self.target_label
        return image, label


def build_model(num_classes: int, freeze_backbone: bool):
    model, _ = _build_mobilenet_v3_small(num_classes=num_classes)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    return model


def get_weights(model) -> List:
    """Model state_dict -> list of numpy ndarrays (Flower's wire format)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def trainable_state_indices(model) -> List[int]:
    """Indices of ``requires_grad`` parameters in ``state_dict`` order (excludes BN buffers)."""
    selected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return [index for index, name in enumerate(model.state_dict()) if name in selected]


def last_layer_state_indices(model) -> List[int]:
    """Indices of the final classifier Linear in ``state_dict`` order."""
    last_ids = {
        id(parameter)
        for parameter in model.classifier[-1].parameters()
        if parameter.requires_grad
    }
    selected = {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in last_ids
    }
    return [index for index, name in enumerate(model.state_dict()) if name in selected]


def set_weights(model, ndarrays: Sequence) -> None:
    """List of numpy ndarrays -> model state_dict (same key order as get_weights)."""
    import torch  # type: ignore
    from collections import OrderedDict

    keys = list(model.state_dict().keys())
    state = OrderedDict(
        (k, torch.tensor(v)) for k, v in zip(keys, ndarrays)
    )
    model.load_state_dict(state, strict=True)


def _make_transforms(img_size: int):
    from torchvision import transforms  # type: ignore

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


def _samples_from_manifest(data_root: Path, manifest: Dict) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for rel_path, label in manifest["samples"]:
        samples.append((data_root / rel_path, int(label)))
    return samples


def _validate_external_eval_manifest(
    partition_manifest: Dict, eval_manifest: Dict
) -> None:
    if eval_manifest.get("split") != "validation":
        raise ValueError(
            "FL round evaluation requires the locked development validation "
            f"manifest, got split={eval_manifest.get('split')!r}"
        )
    if list(partition_manifest["classes"]) != list(eval_manifest["classes"]):
        raise ValueError("Partition and validation manifests use different class orders")
    partition_dataset_id = partition_manifest.get("source_dataset_id")
    if partition_dataset_id is None:
        partition_dataset_id = partition_manifest.get("dataset_id")
    if partition_dataset_id != eval_manifest.get("dataset_id"):
        raise ValueError(
            "Partition and validation manifests belong to different dataset IDs: "
            f"{partition_dataset_id!r} != {eval_manifest.get('dataset_id')!r}"
        )
    partition_paths = {str(path) for path, _label in partition_manifest["samples"]}
    eval_paths = {str(path) for path, _label in eval_manifest["samples"]}
    overlap = partition_paths & eval_paths
    if overlap:
        raise ValueError(
            f"Training partition and validation manifest overlap by {len(overlap)} paths"
        )


def load_data(
    data_root: Path,
    partition_file: Path = None,
    eval_manifest_file: Path = None,
    img_size: int = 224,
    val_split: float = 0.15,
    seed: int = 1337,
    batch_size: int = 16,
    num_workers: int = 2,
    attack_config=None,
):
    """Returns (train_loader, val_loader, classes).

    If ``partition_file`` is given, samples and the global class list come from
    the manifest; otherwise all class folders under ``data_root`` are used
    (single-client / standalone mode).
    """
    from torch.utils.data import DataLoader, Subset  # type: ignore

    partition_manifest = None
    if partition_file is not None:
        manifest = json.loads(Path(partition_file).read_text(encoding="utf-8"))
        partition_manifest = manifest
        classes = list(manifest["classes"])
        samples = _samples_from_manifest(data_root, manifest)
    else:
        classes = _scan_class_folders(data_root)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        samples = _build_samples_from_class_folders(data_root, class_to_idx)

    if len(classes) < 2:
        raise SystemExit(f"Need at least 2 classes, found {len(classes)}")
    if len(samples) < 10:
        raise SystemExit(f"Not enough images: {len(samples)} (need >= 10)")

    train_tf, val_tf = _make_transforms(img_size)
    train_base = _PathsDataset(samples, transform=None)
    if eval_manifest_file is not None:
        if partition_manifest is None:
            raise ValueError("External evaluation requires a partition manifest")
        eval_manifest = json.loads(
            Path(eval_manifest_file).read_text(encoding="utf-8")
        )
        _validate_external_eval_manifest(partition_manifest, eval_manifest)
        eval_samples = _samples_from_manifest(data_root, eval_manifest)
        train_idxs = list(range(len(samples)))
        eval_idxs = list(range(len(eval_samples)))
        eval_base = _PathsDataset(eval_samples, transform=None)
        train_ds = _TransformDataset(Subset(train_base, train_idxs), train_tf)
        val_ds = _TransformDataset(Subset(eval_base, eval_idxs), val_tf)
    else:
        # Legacy mode retained only for reproducing historical runs.
        train_idxs, val_idxs = _split_indices(len(samples), val_split, seed)
        train_ds = _TransformDataset(Subset(train_base, train_idxs), train_tf)
        val_ds = _TransformDataset(Subset(train_base, val_idxs), val_tf)

    if attack_config is not None:
        attack_config.validate_classes(classes)
        flip = attack_config.label_flip
        if flip is not None:
            from app.attacks import select_label_flip_positions

            source_label = classes.index(flip.source_class)
            target_label = classes.index(flip.target_class)
            train_labels = [samples[index][1] for index in train_idxs]
            poisoned_positions = select_label_flip_positions(
                labels=train_labels,
                source_label=source_label,
                fraction=flip.fraction,
                seed=attack_config.seed,
            )
            if not poisoned_positions:
                raise ValueError(
                    f"No {flip.source_class!r} training samples are available for label flipping"
                )
            train_ds = _LabelFlipDataset(
                dataset=train_ds,
                poisoned_positions=poisoned_positions,
                source_label=source_label,
                target_label=target_label,
                source_count=sum(1 for label in train_labels if label == source_label),
                start_round=attack_config.start_round,
                end_round=attack_config.end_round,
            )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    return train_loader, val_loader, classes


def set_training_round(train_loader, server_round: int) -> None:
    """Propagate the current FL round to a round-aware training dataset."""
    setter = getattr(train_loader.dataset, "set_round", None)
    if setter is not None:
        setter(server_round)


def label_flip_metrics(train_loader) -> Dict:
    """Return scalar audit metrics for the configured train-only label flip."""
    dataset = train_loader.dataset
    if not isinstance(dataset, _LabelFlipDataset):
        return {
            "label_flip_attack": False,
            "label_flip_source_count": 0,
            "label_flipped_samples": 0,
        }
    return {
        "label_flip_attack": dataset.is_active(),
        "label_flip_source_count": dataset.source_count,
        "label_flipped_samples": (
            len(dataset.poisoned_positions) if dataset.is_active() else 0
        ),
    }


def train_one_round(
    model,
    train_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    proximal_mu: float = 0.0,
) -> float:
    """Run local epochs and optionally apply the FedProx proximal objective."""
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    device = torch.device("cpu")
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    global_params = [parameter.detach().clone() for parameter in params]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    last_epoch_loss = 0.0
    for _epoch in range(epochs):
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if proximal_mu > 0.0:
                proximal_term = sum(
                    torch.sum((parameter - global_parameter) ** 2)
                    for parameter, global_parameter in zip(params, global_params)
                )
                loss = loss + 0.5 * proximal_mu * proximal_term
            loss.backward()
            optimizer.step()
            bs = int(yb.shape[0])
            running += float(loss.item()) * bs
            seen += bs
        last_epoch_loss = running / max(1, seen)
    return last_epoch_loss


def evaluate_model(model, val_loader) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy) on the local validation split."""
    loss, accuracy, _correct, _total = evaluate_model_detailed(
        model, val_loader, num_classes=None
    )
    return loss, accuracy


def evaluate_model_detailed(model, val_loader, num_classes: Optional[int]):
    """Return loss, accuracy, and clean per-class correct/total counts."""
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    device = torch.device("cpu")
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    class_correct = [0] * int(num_classes or 0)
    class_total = [0] * int(num_classes or 0)
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            predictions = torch.argmax(logits, dim=1)
            loss_sum += float(criterion(logits, yb).item()) * int(yb.shape[0])
            correct += int((predictions == yb).sum().item())
            total += int(yb.shape[0])
            if num_classes is not None:
                for label in range(num_classes):
                    mask = yb == label
                    class_total[label] += int(mask.sum().item())
                    class_correct[label] += int(
                        ((predictions == yb) & mask).sum().item()
                    )
    return (
        loss_sum / max(1, total),
        correct / max(1, total),
        class_correct,
        class_total,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except Exception:
        pass
