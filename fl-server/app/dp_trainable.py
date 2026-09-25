"""Central Gaussian DP applied only to client-declared trainable tensors."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy

from app.dp_accounting import (
    expected_noise_l2,
    isotropic_snr_upper_bound,
    noise_stdv,
)
from app.dp_updates import (
    add_noise_to_trainable,
    clip_trainable_update,
    parse_dp_scope_indices,
)


def parse_trainable_indices(
    results: List[Tuple[ClientProxy, FitRes]], scope: str = "head"
) -> List[int]:
    return parse_dp_scope_indices(
        [fit_res.metrics or {} for _proxy, fit_res in results],
        scope,
    )


class TrainableCentralDP(Strategy):
    """Clip/noise only trainable float tensors, then delegate averaging."""

    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
        seed: int = 1337,
        scope: str = "head",
    ) -> None:
        super().__init__()
        if noise_multiplier < 0:
            raise ValueError("noise_multiplier must be >= 0")
        if clipping_norm <= 0:
            raise ValueError("clipping_norm must be > 0")
        if scope not in ("head", "last"):
            raise ValueError("scope must be 'head' or 'last'")
        self.strategy = strategy
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.num_sampled_clients = num_sampled_clients
        self.scope = scope
        self.rng = np.random.default_rng(seed)
        self.current_round_params: List[np.ndarray] = []
        self.last_trainable_count = 0

    def initialize_parameters(self, client_manager):
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        self.current_round_params = parameters_to_ndarrays(parameters)
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)

    def evaluate(self, server_round, parameters):
        return self.strategy.evaluate(server_round, parameters)

    def aggregate_evaluate(self, server_round, results, failures):
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            return None, {}
        if not results or not self.current_round_params:
            return self.strategy.aggregate_fit(server_round, results, failures)

        client_params = [
            parameters_to_ndarrays(fit_res.parameters) for _proxy, fit_res in results
        ]
        indices = parse_trainable_indices(results, self.scope)
        self.last_trainable_count = len(indices)
        for (_proxy, fit_res), params in zip(results, client_params):
            clipped = clip_trainable_update(
                self.current_round_params,
                params,
                indices,
                self.clipping_norm,
            )
            fit_res.parameters = ndarrays_to_parameters(clipped)

        aggregated, metrics = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        if aggregated is None:
            return None, metrics
        averaged = parameters_to_ndarrays(aggregated)
        noised = add_noise_to_trainable(
            averaged,
            indices,
            self.noise_multiplier,
            self.clipping_norm,
            self.num_sampled_clients,
            self.rng,
        )
        metrics = dict(metrics or {})
        num_params = int(
            sum(int(np.prod(self.current_round_params[index].shape)) for index in indices)
        )
        metrics["dp_trainable_scope"] = self.scope
        metrics["dp_trainable_tensors"] = int(len(indices))
        metrics["dp_trainable_params"] = num_params
        metrics["dp_trainable_indices"] = json.dumps(indices, separators=(",", ":"))
        metrics["dp_noise_stdv"] = float(
            noise_stdv(
                self.noise_multiplier, self.clipping_norm, self.num_sampled_clients
            )
        )
        metrics["dp_expected_noise_l2"] = float(
            expected_noise_l2(
                self.noise_multiplier,
                self.clipping_norm,
                self.num_sampled_clients,
                num_params,
            )
        )
        metrics["dp_snr_upper_bound"] = float(
            isotropic_snr_upper_bound(
                self.noise_multiplier, self.num_sampled_clients, num_params
            )
        )
        return ndarrays_to_parameters(noised), metrics
