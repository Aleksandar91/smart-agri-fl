#!/usr/bin/env bash
# Sequential docker create/start for Flower FL runs.
# Avoids Docker Desktop compose i/o timeouts when starting large PyTorch clients.
set -euo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

NUM_CLIENTS="${NUM_CLIENTS:-5}"
FL_NUM_ROUNDS="${FL_NUM_ROUNDS:-10}"
FL_MIN_CLIENTS="${FL_MIN_CLIENTS:-$NUM_CLIENTS}"
FL_EVAL_CLIENTS="${FL_EVAL_CLIENTS:-$FL_MIN_CLIENTS}"
FL_IMG_SIZE="${FL_IMG_SIZE:-128}"
FL_LOCAL_EPOCHS="${FL_LOCAL_EPOCHS:-1}"
FL_SEED="${FL_SEED:-1337}"
FL_AGGREGATION="${FL_AGGREGATION:-fedavg}"
FL_PROXIMAL_MU="${FL_PROXIMAL_MU:-0.01}"
FL_EXPERIMENT_ID="${FL_EXPERIMENT_ID:-unspecified}"
DATASET_DIR="${DATASET_DIR:-../plantvillage/raw/color}"
PARTITIONS_DIR="${PARTITIONS_DIR:-../docs/experiment_protocol/partitions/pv19-capped/a05/seed-101}"
MODELS_DIR="${MODELS_DIR:-./fl-models}"
EVAL_MANIFEST="${EVAL_MANIFEST:-}"
FL_TRIM_BETA="${FL_TRIM_BETA:-0.25}"
FL_KRUM_MALICIOUS_CLIENTS="${FL_KRUM_MALICIOUS_CLIENTS:-1}"
FL_KRUM_CLIENTS_TO_KEEP="${FL_KRUM_CLIENTS_TO_KEEP:-0}"
FL_DP_NOISE_MULTIPLIER="${FL_DP_NOISE_MULTIPLIER:-0}"
FL_DP_CLIPPING_NORM="${FL_DP_CLIPPING_NORM:-10}"
FL_DP_DELTA="${FL_DP_DELTA:-1e-5}"
FL_DP_CLIP_ONLY="${FL_DP_CLIP_ONLY:-0}"
FL_DP_SCOPE="${FL_DP_SCOPE:-head}"
SERVER_IMAGE="${SERVER_IMAGE:-infra_flower-server}"
CLIENT_IMAGE="${CLIENT_IMAGE:-infra_fl-client-0}"
NETWORK="${NETWORK:-infra_default}"
FL_TLS="${FL_TLS:-0}"
TLS_VOLUME=(-v "$(pwd -W)/certs/out:/certs:ro")
TLS_SERVER_ENV=(-e FL_TLS_CERTS_DIR= -e FL_TLS_REQUIRE_CLIENT_AUTH=1)
TLS_CLIENT_ENV=(-e FL_TLS_CERTS_DIR=)
EVAL_VOLUME=()
EVAL_ENV=(-e FL_EVAL_MANIFEST=)
if [[ -n "$EVAL_MANIFEST" ]]; then
  EVAL_VOLUME=(-v "$(pwd -W)/${EVAL_MANIFEST#./}:/protocol/validation.json:ro")
  EVAL_ENV=(-e FL_EVAL_MANIFEST=/protocol/validation.json)
fi
if [[ "$FL_TLS" != "0" ]]; then
  extra_ip="${LAPTOP_IP:-127.0.0.1}"
  bash "${ROOT}/certs/ensure_mtls.sh" --ips "$extra_ip"
  TLS_SERVER_ENV=(-e FL_TLS_CERTS_DIR=/certs -e FL_TLS_REQUIRE_CLIENT_AUTH=1)
  TLS_CLIENT_ENV=(-e FL_TLS_CERTS_DIR=/certs)
fi

mkdir -p "$MODELS_DIR"
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK"

for name in infra-flower-server-1 \
            infra-fl-client-0-1 infra-fl-client-1-1 infra-fl-client-2-1 \
            infra-fl-client-3-1 infra-fl-client-4-1; do
  docker rm -f "$name" >/dev/null 2>&1 || true
done

docker create --name infra-flower-server-1 --network "$NETWORK" -p 9091:9091 \
  -e FL_LISTEN_ADDRESS=0.0.0.0:9091 \
  -e FL_NUM_ROUNDS="$FL_NUM_ROUNDS" \
  -e FL_MIN_CLIENTS="$FL_MIN_CLIENTS" \
  -e FL_EVAL_CLIENTS="$FL_EVAL_CLIENTS" \
  -e FL_OUTPUT_DIR=/models/global \
  -e FL_EXPERIMENT_ID="$FL_EXPERIMENT_ID" \
  -e FL_AGGREGATION="$FL_AGGREGATION" \
  -e FL_PROXIMAL_MU="$FL_PROXIMAL_MU" \
  -e FL_TRIM_BETA="$FL_TRIM_BETA" \
  -e FL_KRUM_MALICIOUS_CLIENTS="$FL_KRUM_MALICIOUS_CLIENTS" \
  -e FL_KRUM_CLIENTS_TO_KEEP="$FL_KRUM_CLIENTS_TO_KEEP" \
  -e FL_DP_NOISE_MULTIPLIER="$FL_DP_NOISE_MULTIPLIER" \
  -e FL_DP_CLIPPING_NORM="$FL_DP_CLIPPING_NORM" \
  -e FL_DP_DELTA="$FL_DP_DELTA" \
  -e FL_DP_CLIP_ONLY="$FL_DP_CLIP_ONLY" \
  -e FL_DP_SCOPE="$FL_DP_SCOPE" \
  "${TLS_SERVER_ENV[@]}" \
  -v "$(pwd -W)/${MODELS_DIR#./}:/models/global" \
  "${TLS_VOLUME[@]}" \
  "$SERVER_IMAGE" python -m app.fl_server

for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  attack_var="FL_ATTACK_CONFIG_CLIENT_${i}"
  attack_value="${!attack_var:-}"
  extra=()
  if [[ -n "$attack_value" ]]; then
    extra+=(-e "FL_ATTACK_CONFIG=$attack_value")
  fi
  docker create --name "infra-fl-client-${i}-1" --network "$NETWORK" \
    -e FL_SERVER_ADDRESS=infra-flower-server-1:9091 \
    -e FL_DATA_ROOT=/data \
    -e FL_LOCAL_EPOCHS="$FL_LOCAL_EPOCHS" \
    -e FL_BATCH_SIZE=16 \
    -e FL_IMG_SIZE="$FL_IMG_SIZE" \
    -e FL_NUM_WORKERS=0 \
    -e FL_CLIENT_ID="client-${i}" \
    -e FL_SEED="$FL_SEED" \
    -e FL_PARTITION_FILE="/partitions/client_${i}.json" \
    "${EVAL_ENV[@]}" \
    "${TLS_CLIENT_ENV[@]}" \
    "${extra[@]}" \
    -v "$(pwd -W)/${DATASET_DIR#./}:/data:ro" \
    -v "$(pwd -W)/${PARTITIONS_DIR#./}:/partitions:ro" \
    "${EVAL_VOLUME[@]}" \
    -v "$(pwd -W)/attack-configs:/attack-configs:ro" \
    "${TLS_VOLUME[@]}" \
    "$CLIENT_IMAGE" python -m app.fl_client
done

docker start infra-flower-server-1
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  docker start "infra-fl-client-${i}-1"
done
echo "[run_fl_sequential] waiting for ${FL_EXPERIMENT_ID} (${FL_AGGREGATION}, ${NUM_CLIENTS} clients, ${FL_NUM_ROUNDS} rounds)"
status="$(docker wait infra-flower-server-1)"
echo "[run_fl_sequential] server_exit=${status}"
if [[ "${status}" != "0" ]]; then
  echo "[run_fl_sequential] server logs:"
  docker logs infra-flower-server-1 || true
  exit "${status}"
fi
if [[ -n "$EVAL_MANIFEST" ]]; then
  python - "$MODELS_DIR/fl_history.json" "$FL_NUM_ROUNDS" <<'PY'
import json
import sys
from pathlib import Path

history_path = Path(sys.argv[1])
expected_rounds = int(sys.argv[2])
payload = json.loads(history_path.read_text(encoding="utf-8"))
evaluated = {
    int(item["round"])
    for item in payload.get("history", [])
    if item.get("phase") == "evaluate"
    and int(item.get("num_clients", 0)) > 0
    and int(item.get("num_failures", 0)) == 0
}
expected = set(range(1, expected_rounds + 1))
if evaluated != expected:
    raise SystemExit(
        f"Locked validation was not completed for every round: "
        f"expected={sorted(expected)}, evaluated={sorted(evaluated)}"
    )
print(f"[run_fl_sequential] locked validation verified for {expected_rounds} rounds")
PY
fi
