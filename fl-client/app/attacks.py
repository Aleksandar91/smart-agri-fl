"""Reproducible client-side poisoning attacks for FL security experiments.

The module is deliberately independent of Flower and PyTorch so that attack
configuration and update manipulation can be unit-tested cheaply. Raw images
and partition manifests are never modified.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class LabelFlipSpec:
    source_class: str
    target_class: str
    fraction: float


@dataclass(frozen=True)
class ModelUpdateSpec:
    mode: str
    scale: float


@dataclass(frozen=True)
class AttackConfig:
    seed: int
    start_round: int
    end_round: Optional[int]
    label_flip: Optional[LabelFlipSpec]
    model_update: Optional[ModelUpdateSpec]
    source_path: str

    def is_active(self, server_round: int) -> bool:
        if server_round < self.start_round:
            return False
        return self.end_round is None or server_round <= self.end_round

    def attack_types(self, server_round: int) -> List[str]:
        if not self.is_active(server_round):
            return []
        names: List[str] = []
        if self.label_flip is not None:
            names.append("label_flip")
        if self.model_update is not None:
            names.append("model_update")
        return names

    def validate_classes(self, classes: Sequence[str]) -> None:
        if self.label_flip is None:
            return
        known = set(classes)
        for name, value in (
            ("source_class", self.label_flip.source_class),
            ("target_class", self.label_flip.target_class),
        ):
            if value not in known:
                raise ValueError(
                    f"label_flip.{name}={value!r} is not in the global class list"
                )


def _require_mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _reject_unknown_keys(value: dict, allowed: Set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} keys: {unknown}")


def load_attack_config(path: Path) -> AttackConfig:
    """Load and strictly validate one attack configuration JSON file."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Attack config not found: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid attack config JSON in {source}: {exc}") from exc

    raw = _require_mapping(raw, "attack config")
    _reject_unknown_keys(
        raw,
        {"seed", "active_rounds", "label_flip", "model_update", "description"},
        "attack config",
    )

    seed = int(raw.get("seed", 1337))
    rounds = _require_mapping(raw.get("active_rounds", {}), "active_rounds")
    _reject_unknown_keys(rounds, {"start", "end"}, "active_rounds")
    start_round = int(rounds.get("start", 1))
    raw_end = rounds.get("end")
    end_round = int(raw_end) if raw_end is not None else None
    if start_round < 1:
        raise ValueError("active_rounds.start must be >= 1")
    if end_round is not None and end_round < start_round:
        raise ValueError("active_rounds.end must be >= start")

    label_flip = None
    raw_flip = raw.get("label_flip")
    if raw_flip is not None:
        raw_flip = _require_mapping(raw_flip, "label_flip")
        _reject_unknown_keys(
            raw_flip, {"source_class", "target_class", "fraction"}, "label_flip"
        )
        try:
            source_class = str(raw_flip["source_class"])
            target_class = str(raw_flip["target_class"])
        except KeyError as exc:
            raise ValueError(f"Missing label_flip field: {exc.args[0]}") from exc
        fraction = float(raw_flip.get("fraction", 1.0))
        if source_class == target_class:
            raise ValueError("label_flip source_class and target_class must differ")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("label_flip.fraction must be in (0, 1]")
        label_flip = LabelFlipSpec(
            source_class=source_class,
            target_class=target_class,
            fraction=fraction,
        )

    model_update = None
    raw_update = raw.get("model_update")
    if raw_update is not None:
        raw_update = _require_mapping(raw_update, "model_update")
        _reject_unknown_keys(raw_update, {"mode", "scale"}, "model_update")
        mode = str(raw_update.get("mode", "scale_delta"))
        if mode != "scale_delta":
            raise ValueError("model_update.mode must be 'scale_delta'")
        scale = float(raw_update.get("scale", -5.0))
        if not math.isfinite(scale):
            raise ValueError("model_update.scale must be finite")
        model_update = ModelUpdateSpec(mode=mode, scale=scale)

    if label_flip is None and model_update is None:
        raise ValueError("Attack config must define label_flip and/or model_update")

    return AttackConfig(
        seed=seed,
        start_round=start_round,
        end_round=end_round,
        label_flip=label_flip,
        model_update=model_update,
        source_path=str(source),
    )


def select_label_flip_positions(
    labels: Sequence[int],
    source_label: int,
    fraction: float,
    seed: int,
) -> Set[int]:
    """Select deterministic dataset positions to relabel.

    Positions are relative to the local training dataset, not the original
    manifest. A positive fraction always selects at least one source example.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    candidates = [i for i, label in enumerate(labels) if int(label) == source_label]
    if not candidates:
        return set()
    count = max(1, int(math.floor(len(candidates) * fraction)))
    if count >= len(candidates):
        return set(candidates)
    rng = random.Random(seed)
    return set(rng.sample(candidates, count))


def manipulate_model_update(
    global_weights: Sequence,
    local_weights: Sequence,
    config: Optional[AttackConfig],
    server_round: int,
) -> Tuple[List, dict]:
    """Return outgoing weights and auditable update-norm metrics.

    For ``scale_delta``, the malicious client sends
    ``global + scale * (local - global)``. Non-floating state entries (for
    example integer counters) remain unchanged.
    """
    import numpy as np  # type: ignore

    if len(global_weights) != len(local_weights):
        raise ValueError("Global and local weight lists have different lengths")

    attack_active = bool(
        config is not None
        and config.is_active(server_round)
        and config.model_update is not None
    )
    scale = config.model_update.scale if attack_active else 1.0
    outgoing: List = []
    clean_sq = 0.0
    sent_sq = 0.0

    for global_value, local_value in zip(global_weights, local_weights):
        global_array = np.asarray(global_value)
        local_array = np.asarray(local_value)
        if global_array.shape != local_array.shape:
            raise ValueError("Global and local parameter shapes differ")

        if np.issubdtype(local_array.dtype, np.floating):
            delta = local_array.astype(np.float64) - global_array.astype(np.float64)
            clean_sq += float(np.sum(delta * delta))
            sent_delta = scale * delta
            sent_sq += float(np.sum(sent_delta * sent_delta))
            sent = global_array.astype(np.float64) + sent_delta
            if not np.all(np.isfinite(sent)):
                raise ValueError("Poisoned model update contains NaN or infinity")
            outgoing.append(sent.astype(local_array.dtype, copy=False))
        else:
            outgoing.append(local_array.copy())

    metrics = {
        "model_update_attack": attack_active,
        "update_scale": float(scale),
        "clean_delta_l2": math.sqrt(clean_sq),
        "sent_delta_l2": math.sqrt(sent_sq),
    }
    return outgoing, metrics
