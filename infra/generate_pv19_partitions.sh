#!/usr/bin/env bash
# Generate locked PV-19-capped group-safe partitions for the confirmatory matrix.
# Existing partition directories are left untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CLIENT="$(cd "$ROOT/../fl-client" && pwd)"
TRAIN="../docs/experiment_protocol/datasets/pv19-capped-primary-v1/train.json"
OUT_ROOT="../docs/experiment_protocol/partitions/pv19-capped"
SEEDS=(101 211 307 401 503)

generate() {
  local label="$1"
  local mode="$2"
  local alpha="${3:-}"
  local extra=()
  if [[ "$mode" == "dirichlet" ]]; then
    extra+=(--alpha "$alpha")
  fi
  for seed in "${SEEDS[@]}"; do
    local dest="$OUT_ROOT/$label/seed-$seed"
    if [[ -f "$dest/partitions_summary.json" ]]; then
      echo "[partitions] skip existing $label seed=$seed"
      continue
    fi
    echo "[partitions] generating $label seed=$seed"
    mkdir -p "$dest"
    (
      cd "$CLIENT"
      python -m app.partition_dataset \
        --source-manifest "$TRAIN" \
        --output-dir "$dest" \
        --num-clients 5 \
        --mode "$mode" \
        "${extra[@]}" \
        --seed "$seed" \
        --min-per-client 20
    )
  done
}

generate a01 dirichlet 0.1
generate a05 dirichlet 0.5
generate iid iid
echo "[partitions] PV-19-capped partition generation complete"
