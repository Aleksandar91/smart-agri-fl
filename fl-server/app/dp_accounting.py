"""Client-level RDP accountant for Flower central Gaussian DP.

Flower's server-side fixed clipping adds isotropic Gaussian noise with
standard deviation ``noise_multiplier * clipping_norm / n`` to the averaged
update. After clipping each client update to L2 norm ``C``, the average has
sensitivity ``C/n``, so the mechanism is a Gaussian with noise multiplier
``z = noise_multiplier``.

This module accounts **client-level** (farm-level) central DP under full
participation (sampling rate q=1). There is no subsampling amplification.
It does not claim sample-level DP-SGD.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_ALPHAS: Tuple[float, ...] = (
    1.1,
    1.25,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
)


def gaussian_rdp(noise_multiplier: float, alpha: float, steps: int) -> float:
    """RDP of ``steps`` compositions of a Gaussian mechanism with multiplier z."""
    if noise_multiplier <= 0.0:
        return math.inf
    if alpha <= 1.0:
        raise ValueError("RDP order alpha must be > 1")
    if steps < 0:
        raise ValueError("steps must be >= 0")
    if steps == 0:
        return 0.0
    return steps * alpha / (2.0 * noise_multiplier * noise_multiplier)


def rdp_to_epsilon(rdp: float, alpha: float, delta: float) -> float:
    """Convert RDP to (ε, δ)-DP."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if alpha <= 1.0:
        raise ValueError("RDP order alpha must be > 1")
    if not math.isfinite(rdp):
        return math.inf
    return rdp + math.log(1.0 / delta) / (alpha - 1.0)


def account_epsilon(
    noise_multiplier: float,
    steps: int,
    delta: float,
    alphas: Optional[Sequence[float]] = None,
) -> Dict:
    """Return the tightest ε over ``alphas`` for the given Gaussian DP run."""
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError("delta must be in (0, 1)")
    if steps < 0:
        raise ValueError("steps must be >= 0")
    orders: Sequence[float] = tuple(alphas) if alphas is not None else DEFAULT_ALPHAS
    if noise_multiplier <= 0.0:
        return {
            "epsilon": math.inf,
            "delta": delta,
            "alpha": None,
            "noise_multiplier": noise_multiplier,
            "steps": steps,
            "sampling_rate": 1.0,
        }

    best_eps = math.inf
    best_alpha: Optional[float] = None
    for alpha in orders:
        eps = rdp_to_epsilon(
            gaussian_rdp(noise_multiplier, float(alpha), steps),
            float(alpha),
            delta,
        )
        if eps < best_eps:
            best_eps = eps
            best_alpha = float(alpha)
    return {
        "epsilon": best_eps,
        "delta": delta,
        "alpha": best_alpha,
        "noise_multiplier": noise_multiplier,
        "steps": steps,
        "sampling_rate": 1.0,
    }


def noise_stdv(noise_multiplier: float, clipping_norm: float, num_clients: int) -> float:
    """Match Flower ``compute_stdv``: σ = z · C / n on the averaged update."""
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    return noise_multiplier * clipping_norm / float(num_clients)


def expected_noise_l2(
    noise_multiplier: float,
    clipping_norm: float,
    num_clients: int,
    num_params: int,
) -> float:
    """Expected L2 of isotropic Gaussian noise in ``num_params`` dimensions."""
    if num_params < 0:
        raise ValueError("num_params must be >= 0")
    return noise_stdv(noise_multiplier, clipping_norm, num_clients) * math.sqrt(
        num_params
    )


def isotropic_snr_upper_bound(
    noise_multiplier: float, num_clients: int, num_params: int
) -> float:
    """Upper bound on ||clipped avg update|| / ||noise||.

    After per-client L2 clipping to C, the averaged update has L2 at most C.
    Isotropic Gaussian noise with σ = z·C/n has expected L2 σ√d. The ratio
    ``n / (z √d)`` does **not** depend on C: tightening the clip shrinks
    signal and noise together.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if num_params < 0:
        raise ValueError("num_params must be >= 0")
    if noise_multiplier <= 0.0 or num_params == 0:
        return math.inf
    return num_clients / (noise_multiplier * math.sqrt(num_params))


def account_report(
    noise_multiplier: float,
    steps: int,
    clipping_norm: float,
    num_clients: int,
    deltas: Sequence[float] = (1e-5, 1e-3),
) -> Dict:
    """Bundle accounting used in experiment metadata."""
    spent: List[Dict] = [
        account_epsilon(noise_multiplier, steps, float(delta)) for delta in deltas
    ]
    return {
        "mechanism": "central_gaussian_client_level_trainable_tensors",
        "noise_multiplier": noise_multiplier,
        "clipping_norm": clipping_norm,
        "num_sampled_clients": num_clients,
        "rounds": steps,
        "sampling_rate": 1.0,
        "noise_stdv": noise_stdv(noise_multiplier, clipping_norm, num_clients),
        "spent": spent,
    }
