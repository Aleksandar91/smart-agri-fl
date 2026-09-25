"""Numpy helpers for trainable-only central DP (Flower-free)."""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from app.dp_accounting import noise_stdv

FROZEN_L2_EPS = 1e-8


def parse_index_list_json(
    metrics_list: List[Dict], key: str, empty_message: str
) -> List[int]:
    """Require every client to report the same integer index list under ``key``."""
    parsed: List[List[int]] = []
    for metrics in metrics_list:
        raw = (metrics or {}).get(key)
        if raw in (None, ""):
            raise ValueError(empty_message)
        parsed.append([int(value) for value in json.loads(str(raw))])
    if not parsed:
        raise ValueError(f"No client results from which to read {key}")
    first = parsed[0]
    if not first:
        raise ValueError(f"{key} is empty")
    if any(item != first for item in parsed):
        raise ValueError(f"Clients reported different {key} values")
    return first


def parse_trainable_indices_json(metrics_list: List[Dict]) -> List[int]:
    return parse_index_list_json(
        metrics_list,
        "trainable_indices_json",
        "Client omitted trainable_indices_json; refusing the L2-heuristic "
        "fallback that previously noised frozen backbone tensors",
    )


def parse_dp_scope_indices(metrics_list: List[Dict], scope: str) -> List[int]:
    """Select the DP subspace: full trainable head, or last Linear only."""
    if scope not in ("head", "last"):
        raise ValueError("dp scope must be 'head' or 'last'")
    head = parse_trainable_indices_json(metrics_list)
    if scope == "head":
        return head
    last = parse_index_list_json(
        metrics_list,
        "last_layer_indices_json",
        "Client omitted last_layer_indices_json required for last-layer DP",
    )
    if not set(last).issubset(set(head)):
        raise ValueError("last-layer indices are not a subset of trainable indices")
    return last


def _is_float_array(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.floating)


def trainable_indices(
    current: List[np.ndarray],
    client_params: List[List[np.ndarray]],
    frozen_l2_eps: float = FROZEN_L2_EPS,
) -> List[int]:
    if not client_params:
        return []
    selected = []
    for index in range(len(current)):
        if not _is_float_array(current[index]):
            continue
        max_l2 = 0.0
        for params in client_params:
            delta = params[index].astype(np.float64, copy=False) - current[index].astype(
                np.float64, copy=False
            )
            max_l2 = max(max_l2, float(np.linalg.norm(delta)))
        if max_l2 > frozen_l2_eps:
            selected.append(index)
    return selected


def clip_trainable_update(
    current: List[np.ndarray],
    updated: List[np.ndarray],
    indices: List[int],
    clipping_norm: float,
) -> List[np.ndarray]:
    if not indices:
        return [array.copy() for array in current]
    flats = []
    shapes = []
    dtypes = []
    for index in indices:
        delta = updated[index].astype(np.float64, copy=False) - current[index].astype(
            np.float64, copy=False
        )
        shapes.append(delta.shape)
        dtypes.append(updated[index].dtype)
        flats.append(delta.ravel())
    stacked = np.concatenate(flats)
    norm = float(np.linalg.norm(stacked))
    scale = clipping_norm / norm if norm > clipping_norm and norm > 0.0 else 1.0
    stacked = stacked * scale
    rebuilt = [array.copy() for array in updated]
    offset = 0
    for index, shape, dtype in zip(indices, shapes, dtypes):
        size = int(np.prod(shape))
        piece = stacked[offset : offset + size].reshape(shape)
        rebuilt[index] = (current[index].astype(np.float64) + piece).astype(
            dtype, copy=False
        )
        offset += size
    return rebuilt


def add_noise_to_trainable(
    averaged: List[np.ndarray],
    indices: List[int],
    noise_multiplier: float,
    clipping_norm: float,
    num_clients: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    if not indices:
        return [array.copy() for array in averaged]
    stdv = noise_stdv(noise_multiplier, clipping_norm, num_clients)
    noised = [array.copy() for array in averaged]
    for index in indices:
        noise = rng.normal(0.0, stdv, size=averaged[index].shape)
        noised[index] = (averaged[index].astype(np.float64) + noise).astype(
            averaged[index].dtype, copy=False
        )
    return noised
