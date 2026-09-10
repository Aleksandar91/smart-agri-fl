"""Flower federated learning server (FedAvg; optional gRPC mTLS).

Runs a gRPC server (default port 9091) and coordinates FL rounds:
  round = sample clients -> clients train locally -> FedAvg aggregation
          -> clients evaluate the new global model.

Design notes:
  - This process is torch-free on purpose: FedAvg only averages numpy arrays,
    and the initial parameters are requested from a randomly picked client.
    The server image therefore stays small (flwr + numpy only).
  - After each round the aggregated global model is saved to OUTPUT_DIR as
    a .npz (list of arrays, same order as the client's state_dict), plus a
    metrics history JSON - enough to reconstruct the model on any client
    with fl_task.set_weights().

Run:
    python -m app.fl_server --rounds 10 --min-clients 2

The FastAPI health app (app.main) keeps running separately on port 8080.
"""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from app.dp_accounting import account_report


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


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw != "" else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    parser = argparse.ArgumentParser(description="Flower FL server (FedAvg).")
    parser.add_argument(
        "--listen-address",
        type=str,
        default=_env_str("FL_LISTEN_ADDRESS", "0.0.0.0:9091"),
        help="host:port for the Flower gRPC server (env: FL_LISTEN_ADDRESS).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=_env_int("FL_NUM_ROUNDS", 10),
        help="Number of federated rounds (env: FL_NUM_ROUNDS).",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=_env_int("FL_MIN_CLIENTS", 2),
        help="Minimum clients required for training/evaluation (env: FL_MIN_CLIENTS).",
    )
    parser.add_argument(
        "--eval-clients",
        type=int,
        default=_env_int("FL_EVAL_CLIENTS", 0),
        help=(
            "Clients sampled for round evaluation; 0 uses all min-clients "
            "(env: FL_EVAL_CLIENTS). Use 1 when all clients share one locked validation set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_env_str("FL_OUTPUT_DIR", "/models/global"),
        help="Where to save global model .npz + metrics (env: FL_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=_env_str("FL_EXPERIMENT_ID", "unspecified"),
        help="Stable identifier stored in fit/evaluation history.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["fedavg", "fedprox", "median", "trimmed_mean", "krum"],
        default=_env_str("FL_AGGREGATION", "fedavg"),
        help="Server aggregation rule (env: FL_AGGREGATION).",
    )
    parser.add_argument(
        "--proximal-mu",
        type=float,
        default=_env_float("FL_PROXIMAL_MU", 0.01),
        help="FedProx proximal coefficient mu (env: FL_PROXIMAL_MU).",
    )
    parser.add_argument(
        "--trim-beta",
        type=float,
        default=_env_float("FL_TRIM_BETA", 0.25),
        help="Fraction removed from each tail by trimmed mean.",
    )
    parser.add_argument(
        "--krum-malicious-clients",
        type=int,
        default=_env_int("FL_KRUM_MALICIOUS_CLIENTS", 1),
        help="Assumed Byzantine client count for Krum.",
    )
    parser.add_argument(
        "--krum-clients-to-keep",
        type=int,
        default=_env_int("FL_KRUM_CLIENTS_TO_KEEP", 0),
        help="0 for Krum; >0 for MultiKrum.",
    )
    parser.add_argument(
        "--dp-noise-multiplier",
        type=float,
        default=_env_float("FL_DP_NOISE_MULTIPLIER", 0.0),
        help="Central Gaussian DP noise multiplier z; 0 disables DP.",
    )
    parser.add_argument(
        "--dp-clipping-norm",
        type=float,
        default=_env_float("FL_DP_CLIPPING_NORM", 10.0),
        help="L2 clipping norm C for client updates (env: FL_DP_CLIPPING_NORM).",
    )
    parser.add_argument(
        "--dp-delta",
        type=float,
        default=_env_float("FL_DP_DELTA", 1e-5),
        help="Primary δ for (ε, δ)-DP reporting (env: FL_DP_DELTA).",
    )
    parser.add_argument(
        "--dp-clip-only",
        action="store_true",
        default=_env_bool("FL_DP_CLIP_ONLY", False),
        help="Apply trainable L2 clipping without Gaussian noise (env: FL_DP_CLIP_ONLY).",
    )
    parser.add_argument(
        "--dp-scope",
        choices=["head", "last"],
        default=_env_str("FL_DP_SCOPE", "head"),
        help="Clip/noise the full trainable head or only the last Linear (env: FL_DP_SCOPE).",
    )
    args = parser.parse_args()
    if args.eval_clients == 0:
        args.eval_clients = args.min_clients
    if not 1 <= args.eval_clients <= args.min_clients:
        raise SystemExit("--eval-clients must be between 1 and --min-clients")
    if not 0.0 <= args.trim_beta < 0.5:
        raise SystemExit("--trim-beta must be in [0, 0.5)")
    if args.krum_malicious_clients < 0:
        raise SystemExit("--krum-malicious-clients must be >= 0")
    if args.krum_clients_to_keep < 0:
        raise SystemExit("--krum-clients-to-keep must be >= 0")
    if args.dp_noise_multiplier < 0:
        raise SystemExit("--dp-noise-multiplier must be >= 0")
    if args.dp_clipping_norm <= 0:
        raise SystemExit("--dp-clipping-norm must be > 0")
    if not 0.0 < args.dp_delta < 1.0:
        raise SystemExit("--dp-delta must be in (0, 1)")
    if args.dp_clip_only and args.dp_noise_multiplier > 0:
        raise SystemExit("--dp-clip-only requires --dp-noise-multiplier 0")
    if args.proximal_mu < 0:
        raise SystemExit("--proximal-mu must be >= 0")
    if (
        (args.dp_noise_multiplier > 0 or args.dp_clip_only)
        and args.aggregation not in ("fedavg", "krum")
    ):
        raise SystemExit(
            "Central DP is implemented around FedAvg and Krum/MultiKrum only; "
            "FedProx, median and trimmed mean remain non-private in this phase"
        )
    if args.aggregation == "krum":
        required_clients = 2 * args.krum_malicious_clients + 3
        if args.min_clients < required_clients:
            raise SystemExit(
                "Krum requires at least 2f+3 clients for the configured threat "
                f"model: f={args.krum_malicious_clients}, need "
                f"{required_clients}, got {args.min_clients}"
            )

    import numpy as np  # type: ignore
    import flwr as fl  # type: ignore
    from flwr.common import parameters_to_ndarrays  # type: ignore
    from app.tls_runtime import load_server_certificates
    from flwr.server.strategy import (  # type: ignore
        FedAvg,
        FedProx,
        FedMedian,
        FedTrimmedAvg,
        Krum,
    )
    from flwr.server.strategy.strategy import Strategy  # type: ignore
    from app.dp_trainable import TrainableCentralDP

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict] = []
    run_started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _fit_config(server_round: int) -> Dict:
        config = {
            "server_round": server_round,
            "experiment_id": args.experiment_id,
        }
        if args.aggregation == "fedprox":
            config["proximal_mu"] = float(args.proximal_mu)
        return config

    def _weighted_avg_acc(metrics: List[Tuple[int, Dict]]) -> Dict:
        total = sum(n for n, _m in metrics)
        if total == 0:
            return {}
        acc = sum(n * float(m.get("val_acc", 0.0)) for n, m in metrics) / total
        aggregated = {"val_acc": acc}

        class_names = None
        class_correct = None
        class_total = None
        for _n, client_metrics in metrics:
            if not all(
                key in client_metrics
                for key in (
                    "class_names_json",
                    "class_correct_json",
                    "class_total_json",
                )
            ):
                class_names = None
                break
            names = json.loads(str(client_metrics["class_names_json"]))
            correct = [
                int(value)
                for value in json.loads(str(client_metrics["class_correct_json"]))
            ]
            totals = [
                int(value)
                for value in json.loads(str(client_metrics["class_total_json"]))
            ]
            if not (len(names) == len(correct) == len(totals)):
                raise ValueError("Client per-class metric lengths do not match")
            if class_names is None:
                class_names = names
                class_correct = [0] * len(names)
                class_total = [0] * len(names)
            elif names != class_names:
                raise ValueError("Clients reported different global class orders")
            for index in range(len(names)):
                class_correct[index] += correct[index]
                class_total[index] += totals[index]

        if class_names is not None:
            per_class = {
                name: (
                    class_correct[index] / class_total[index]
                    if class_total[index] > 0
                    else None
                )
                for index, name in enumerate(class_names)
            }
            aggregated["per_class_acc_json"] = json.dumps(
                per_class, separators=(",", ":")
            )
            aggregated["per_class_total_json"] = json.dumps(
                dict(zip(class_names, class_total)), separators=(",", ":")
            )
        return aggregated

    def _fit_client_audit(results) -> List[Dict]:
        allowed = (
            "client_id",
            "experiment_id",
            "train_loss",
            "local_epochs",
            "seed",
            "source_dataset_id",
            "source_dataset_manifest_sha256",
            "proximal_mu",
            "attack_active",
            "attack_types",
            "label_flip_attack",
            "label_flip_source_count",
            "label_flipped_samples",
            "model_update_attack",
            "update_scale",
            "clean_delta_l2",
            "sent_delta_l2",
            "trainable_indices_json",
            "last_layer_indices_json",
        )
        audit: List[Dict] = []
        for _client_proxy, fit_result in results:
            source = fit_result.metrics or {}
            item = {}
            for key in allowed:
                if key not in source:
                    continue
                value = source[key]
                if isinstance(value, float) and not math.isfinite(value):
                    value = None
                item[key] = value
            item["num_examples"] = int(fit_result.num_examples)
            audit.append(item)
        audit.sort(key=lambda item: str(item.get("client_id", "")))
        return audit

    strategy_classes = {
        "fedavg": FedAvg,
        "fedprox": FedProx,
        "median": FedMedian,
        "trimmed_mean": FedTrimmedAvg,
        "krum": Krum,
    }
    strategy_base = strategy_classes[args.aggregation]
    dp_enabled = args.dp_noise_multiplier > 0.0 or args.dp_clip_only
    dp_config = (
        {
            **account_report(
                args.dp_noise_multiplier,
                args.rounds,
                args.dp_clipping_norm,
                args.min_clients,
                deltas=(args.dp_delta, 1e-3),
            ),
            "clip_only": bool(args.dp_clip_only),
            "trainable_scope": args.dp_scope,
        }
        if dp_enabled
        else None
    )

    class SavingStrategy(Strategy):
        """Persist checkpoints after the inner strategy (including DP noise)."""

        def __init__(self, inner: Strategy):
            super().__init__()
            self.inner = inner

        def initialize_parameters(self, client_manager):
            return self.inner.initialize_parameters(client_manager)

        def configure_fit(self, server_round, parameters, client_manager):
            instructions = self.inner.configure_fit(
                server_round, parameters, client_manager
            )
            if args.aggregation == "fedprox":
                for _client, fit_ins in instructions:
                    fit_ins.config["proximal_mu"] = float(args.proximal_mu)
            return instructions

        def configure_evaluate(self, server_round, parameters, client_manager):
            return self.inner.configure_evaluate(
                server_round, parameters, client_manager
            )

        def evaluate(self, server_round, parameters):
            return self.inner.evaluate(server_round, parameters)

        def aggregate_fit(self, server_round, results, failures):
            t0 = time.time()
            parameters, metrics = self.inner.aggregate_fit(
                server_round, results, failures
            )
            if parameters is not None:
                ndarrays = parameters_to_ndarrays(parameters)
                npz_path = out_dir / f"global_round_{server_round:03d}.npz"
                np.savez(npz_path, *ndarrays)
                np.savez(out_dir / "global_latest.npz", *ndarrays)
                num_examples = sum(r.num_examples for _c, r in results)
                rec = {
                    "round": server_round,
                    "phase": "fit",
                    "aggregation": args.aggregation,
                    "num_clients": len(results),
                    "num_failures": len(failures),
                    "num_examples": num_examples,
                    "client_metrics": _fit_client_audit(results),
                    "aggregate_took_s": round(time.time() - t0, 3),
                    "saved": npz_path.name,
                    "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                if dp_enabled:
                    rec["dp_noise_stdv"] = dp_config["noise_stdv"]
                    if metrics:
                        rec["dp_trainable_scope"] = metrics.get("dp_trainable_scope")
                        rec["dp_trainable_tensors"] = metrics.get(
                            "dp_trainable_tensors"
                        )
                        rec["dp_trainable_params"] = metrics.get("dp_trainable_params")
                        rec["dp_trainable_indices"] = metrics.get(
                            "dp_trainable_indices"
                        )
                        rec["dp_expected_noise_l2"] = metrics.get(
                            "dp_expected_noise_l2"
                        )
                        rec["dp_snr_upper_bound"] = metrics.get("dp_snr_upper_bound")
                history.append(rec)
                _flush_history()
                print("[fl-server] fit_aggregated", rec)
            return parameters, metrics

        def aggregate_evaluate(self, server_round, results, failures):
            loss, metrics = self.inner.aggregate_evaluate(
                server_round, results, failures
            )
            rec = {
                "round": server_round,
                "phase": "evaluate",
                "num_clients": len(results),
                "num_failures": len(failures),
                "global_val_loss": (
                    float(loss)
                    if loss is not None and math.isfinite(float(loss))
                    else None
                ),
                "global_val_acc": (
                    float(metrics["val_acc"])
                    if "val_acc" in metrics
                    and math.isfinite(float(metrics["val_acc"]))
                    else None
                ),
                "per_class_acc": (
                    json.loads(str(metrics["per_class_acc_json"]))
                    if "per_class_acc_json" in metrics
                    else None
                ),
                "per_class_total": (
                    json.loads(str(metrics["per_class_total_json"]))
                    if "per_class_total_json" in metrics
                    else None
                ),
                "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            history.append(rec)
            _flush_history()
            print("[fl-server] evaluate_aggregated", rec)
            return loss, metrics

    def _json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: _json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return value

    def _flush_history() -> None:
        (out_dir / "fl_history.json").write_text(
            json.dumps(
                _json_safe(
                    {
                        "experiment_id": args.experiment_id,
                    "aggregation": args.aggregation,
                    "rounds": args.rounds,
                    "min_clients": args.min_clients,
                    "eval_clients": args.eval_clients,
                    "aggregation_config": {
                        "proximal_mu": (
                            args.proximal_mu
                            if args.aggregation == "fedprox"
                            else None
                        ),
                        "trim_beta": (
                            args.trim_beta
                            if args.aggregation == "trimmed_mean"
                            else None
                        ),
                        "krum_malicious_clients": (
                            args.krum_malicious_clients
                            if args.aggregation == "krum"
                            else None
                        ),
                        "krum_clients_to_keep": (
                            args.krum_clients_to_keep
                            if args.aggregation == "krum"
                            else None
                        ),
                    },
                    "dp_config": dp_config,
                    "run_started_utc": run_started,
                    "history": history,
                    }
                ),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    strategy_kwargs = {
        "fraction_fit": 1.0,
        "fraction_evaluate": args.eval_clients / args.min_clients,
        "min_fit_clients": args.min_clients,
        "min_evaluate_clients": args.eval_clients,
        "min_available_clients": args.min_clients,
        "on_fit_config_fn": _fit_config,
        "on_evaluate_config_fn": _fit_config,
        "evaluate_metrics_aggregation_fn": _weighted_avg_acc,
    }
    if args.aggregation == "trimmed_mean":
        strategy_kwargs["beta"] = args.trim_beta
    elif args.aggregation == "fedprox":
        strategy_kwargs["proximal_mu"] = args.proximal_mu
    elif args.aggregation == "krum":
        strategy_kwargs["num_malicious_clients"] = args.krum_malicious_clients
        strategy_kwargs["num_clients_to_keep"] = args.krum_clients_to_keep
    inner_strategy = strategy_base(**strategy_kwargs)
    if dp_enabled:
        inner_strategy = TrainableCentralDP(
            inner_strategy,
            args.dp_noise_multiplier,
            args.dp_clipping_norm,
            args.min_clients,
            seed=1337,
            scope=args.dp_scope,
        )
    strategy = SavingStrategy(inner_strategy)

    tls_certificates = load_server_certificates()
    print(
        "[fl-server] starting",
        {
            "listen": args.listen_address,
            "rounds": args.rounds,
            "min_clients": args.min_clients,
            "eval_clients": args.eval_clients,
            "output_dir": str(out_dir),
            "experiment_id": args.experiment_id,
            "aggregation": args.aggregation,
            "proximal_mu": args.proximal_mu
            if args.aggregation == "fedprox"
            else None,
            "dp": dp_config,
            "tls": "mtls" if tls_certificates is not None else "plaintext",
        },
    )

    fl.server.start_server(
        server_address=args.listen_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        certificates=tls_certificates,
    )

    print("[fl-server] done", {"rounds": args.rounds, "output_dir": str(out_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
