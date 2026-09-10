"""Flower federated learning client (FedAvg; optional gRPC mTLS).

Connects to the Flower server (gRPC), and in each round:
  1. receives global model weights,
  2. trains locally for FL_LOCAL_EPOCHS on its own data partition,
  3. sends updated weights + sample count back (server does FedAvg),
  4. evaluates the (local) model on its local validation split.

Raw images NEVER leave this machine - only model weights do.

Run (standalone, laptop):
    python -m app.fl_client --server-address localhost:9091 --data-root /data \
        --partition-file /partitions/client_0.json

Run (Docker): see infra/docker-compose.yml, profile "fl".
"""

import argparse
import json
import os
import socket
from pathlib import Path

from app import attacks, fl_task


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
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Flower FL client (plant disease).")
    parser.add_argument(
        "--server-address",
        type=str,
        default=_env_str("FL_SERVER_ADDRESS", "localhost:9091"),
        help="Flower server host:port (env: FL_SERVER_ADDRESS).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=_env_str("FL_DATA_ROOT", "/data"),
        help="Dataset root with class subfolders (env: FL_DATA_ROOT).",
    )
    parser.add_argument(
        "--partition-file",
        type=str,
        default=_env_str("FL_PARTITION_FILE", ""),
        help="Optional partition manifest JSON from partition_dataset.py (env: FL_PARTITION_FILE).",
    )
    parser.add_argument(
        "--eval-manifest",
        type=str,
        default=_env_str("FL_EVAL_MANIFEST", ""),
        help=(
            "Locked global development-validation manifest used for round evaluation "
            "(env: FL_EVAL_MANIFEST)."
        ),
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=_env_str("FL_CLIENT_ID", socket.gethostname()),
        help="Client identifier for logs (env: FL_CLIENT_ID).",
    )
    parser.add_argument("--local-epochs", type=int, default=_env_int("FL_LOCAL_EPOCHS", 1))
    parser.add_argument("--batch-size", type=int, default=_env_int("FL_BATCH_SIZE", 16))
    parser.add_argument("--img-size", type=int, default=_env_int("FL_IMG_SIZE", 224))
    parser.add_argument("--lr", type=float, default=_env_float("FL_LR", 1e-3))
    parser.add_argument("--weight-decay", type=float, default=_env_float("FL_WEIGHT_DECAY", 1e-4))
    parser.add_argument("--val-split", type=float, default=_env_float("FL_VAL_SPLIT", 0.15))
    parser.add_argument("--seed", type=int, default=_env_int("FL_SEED", 1337))
    parser.add_argument("--num-workers", type=int, default=_env_int("FL_NUM_WORKERS", 2))
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=_env_bool("FL_FREEZE_BACKBONE", True),
        help="Train only the classifier head (CPU-friendly, smaller updates).",
    )
    parser.add_argument(
        "--attack-config",
        type=str,
        default=_env_str("FL_ATTACK_CONFIG", ""),
        help="Optional poisoning attack JSON (env: FL_ATTACK_CONFIG).",
    )
    args = parser.parse_args()

    import flwr as fl  # type: ignore
    from app.tls_runtime import prepare_client_tls

    fl_task.seed_everything(args.seed)
    tls_root = prepare_client_tls(args.client_id)

    data_root = Path(args.data_root).expanduser()
    partition_file = Path(args.partition_file).expanduser() if args.partition_file else None
    eval_manifest_file = (
        Path(args.eval_manifest).expanduser() if args.eval_manifest else None
    )
    attack_config = None
    partition_metadata = (
        json.loads(partition_file.read_text(encoding="utf-8"))
        if partition_file is not None
        else {}
    )
    if args.attack_config:
        try:
            attack_config = attacks.load_attack_config(
                Path(args.attack_config).expanduser()
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid attack configuration: {exc}") from exc

    train_loader, val_loader, classes = fl_task.load_data(
        data_root=data_root,
        partition_file=partition_file,
        eval_manifest_file=eval_manifest_file,
        img_size=args.img_size,
        val_split=args.val_split,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        attack_config=attack_config,
    )
    model = fl_task.build_model(num_classes=len(classes), freeze_backbone=args.freeze_backbone)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(
        "[fl-client] starting",
        {
            "client_id": args.client_id,
            "server": args.server_address,
            "data_root": str(data_root),
            "partition_file": str(partition_file) if partition_file else None,
            "eval_manifest": str(eval_manifest_file)
            if eval_manifest_file
            else None,
            "classes": len(classes),
            "train_samples": n_train,
            "val_samples": n_val,
            "local_epochs": args.local_epochs,
            "freeze_backbone": args.freeze_backbone,
            "attack_config": attack_config.source_path if attack_config else None,
            "tls": "mtls" if tls_root is not None else "plaintext",
        },
    )

    class PlantDiseaseClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return fl_task.get_weights(model)

        def fit(self, parameters, config):
            import numpy as np  # type: ignore

            server_round = int(config.get("server_round", -1))
            proximal_mu = float(config.get("proximal_mu", 0.0))
            if proximal_mu < 0.0:
                raise ValueError("proximal_mu must be >= 0")
            global_weights = [np.array(value, copy=True) for value in parameters]
            fl_task.set_weights(model, parameters)
            fl_task.set_training_round(train_loader, server_round)
            train_loss = fl_task.train_one_round(
                model,
                train_loader,
                epochs=args.local_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                proximal_mu=proximal_mu,
            )
            local_weights = fl_task.get_weights(model)
            outgoing_weights, update_metrics = attacks.manipulate_model_update(
                global_weights=global_weights,
                local_weights=local_weights,
                config=attack_config,
                server_round=server_round,
            )
            flip_metrics = fl_task.label_flip_metrics(train_loader)
            attack_types = (
                attack_config.attack_types(server_round) if attack_config else []
            )
            fit_metrics = {
                "train_loss": float(train_loss),
                "client_id": args.client_id,
                "experiment_id": str(config.get("experiment_id", "")),
                "local_epochs": int(args.local_epochs),
                "seed": int(args.seed),
                "source_dataset_id": str(
                    partition_metadata.get("source_dataset_id", "")
                ),
                "source_dataset_manifest_sha256": str(
                    partition_metadata.get("source_dataset_manifest_sha256", "")
                ),
                "proximal_mu": proximal_mu,
                "attack_active": bool(attack_types),
                "attack_types": ",".join(attack_types),
                "trainable_indices_json": json.dumps(
                    fl_task.trainable_state_indices(model), separators=(",", ":")
                ),
                "last_layer_indices_json": json.dumps(
                    fl_task.last_layer_state_indices(model), separators=(",", ":")
                ),
                **flip_metrics,
                **update_metrics,
            }
            print(
                "[fl-client] fit_done",
                {
                    "client_id": args.client_id,
                    "round": server_round,
                    "train_loss": round(train_loss, 4),
                    "attack_active": fit_metrics["attack_active"],
                    "attack_types": fit_metrics["attack_types"],
                    "label_flipped_samples": fit_metrics["label_flipped_samples"],
                    "update_scale": fit_metrics["update_scale"],
                    "clean_delta_l2": round(fit_metrics["clean_delta_l2"], 4),
                    "sent_delta_l2": round(fit_metrics["sent_delta_l2"], 4),
                },
            )
            return (
                outgoing_weights,
                n_train,
                fit_metrics,
            )

        def evaluate(self, parameters, config):
            server_round = int(config.get("server_round", -1))
            fl_task.set_weights(model, parameters)
            loss, acc, class_correct, class_total = fl_task.evaluate_model_detailed(
                model, val_loader, num_classes=len(classes)
            )
            print(
                "[fl-client] evaluate_done",
                {
                    "client_id": args.client_id,
                    "round": server_round,
                    "val_loss": round(loss, 4),
                    "val_acc": round(acc, 4),
                },
            )
            return (
                loss,
                n_val,
                {
                    "val_acc": acc,
                    "client_id": args.client_id,
                    "class_names_json": json.dumps(classes, separators=(",", ":")),
                    "class_correct_json": json.dumps(
                        class_correct, separators=(",", ":")
                    ),
                    "class_total_json": json.dumps(
                        class_total, separators=(",", ":")
                    ),
                },
            )

    start_kwargs = {}
    if tls_root is not None:
        start_kwargs["root_certificates"] = tls_root
        start_kwargs["insecure"] = False
    fl.client.start_client(
        server_address=args.server_address,
        client=PlantDiseaseClient().to_client(),
        **start_kwargs,
    )
    print("[fl-client] finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
