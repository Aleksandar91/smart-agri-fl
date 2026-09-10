#!/usr/bin/env bash
# PV-19-capped stage 4.2c: model-update scale -0.5 and -1.0.
# Attacker = locked Apple___healthy monopoly client for that partition.
set -euo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${PV19_MU_MODE:-list}"
ONLY="${PV19_MU_ONLY:-}"
SEEDS=(101 211 307 401 503)
ALPHAS=(a01 a05 iid)
ALGOS=(fedavg fedprox)
SCALES=(minus05 minus1)

RAW_COLOR="${PV19_RAW_COLOR:-../plantvillage/raw/color}"
VAL_MANIFEST="../docs/experiment_protocol/datasets/pv19-capped-primary-v1/validation.json"
RUNS_ROOT="../docs/experiment_protocol/runs/pv19-capped-model-update"
ATTACKERS="../docs/experiment_protocol/attack_attacker_clients.json"
REGISTRY="$RUNS_ROOT/model_update_jobs.json"
LOG="$RUNS_ROOT/phase4_model_update.log"

scale_config() {
  case "$1" in
    minus05) echo "/attack-configs/model_update_scale_minus05.json" ;;
    minus1) echo "/attack-configs/model_update_scale_minus1.json" ;;
    *) echo "unknown" ;;
  esac
}

scale_value() {
  case "$1" in
    minus05) echo "-0.5" ;;
    minus1) echo "-1.0" ;;
    *) echo "unknown" ;;
  esac
}

job_name() {
  echo "pv19-capped-${1}-s${2}-${3}-e1-mu${4}"
}

completed() {
  local history="$1/fl_history.json"
  [[ -f "$history" ]] || return 1
  python - "$history" "${FL_NUM_ROUNDS:-10}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = set(range(1, int(sys.argv[2]) + 1))
evaluated = {
    int(item["round"])
    for item in payload.get("history", [])
    if item.get("phase") == "evaluate"
    and int(item.get("num_clients", 0)) > 0
    and int(item.get("num_failures", 0)) == 0
}
raise SystemExit(0 if evaluated == expected else 1)
PY
}

attacker_client() {
  python - "$ATTACKERS" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
alpha, seed = sys.argv[2], sys.argv[3]
for pair in payload["pairs"]:
    if pair["source"] == "Apple___healthy":
        print(pair["attackers"][alpha][seed])
        break
else:
    raise SystemExit("apple attacker missing")
PY
}

evaluate_validation() {
  local models_dir="$1"
  MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$(pwd -W)/${RAW_COLOR#./}:/data:ro" \
    -v "$(pwd -W)/../docs/experiment_protocol/datasets/pv19-capped-primary-v1:/protocol:ro" \
    -v "$(pwd -W)/${models_dir#./}:/out" \
    infra_fl-client-0 python -m app.evaluate_global \
      --data-root /data \
      --manifest /protocol/validation.json \
      --checkpoint /out/global_latest.npz \
      --output-json /out/validation_evaluation.json \
      --img-size 128 --batch-size 32 --num-workers 0
}

evaluate_split() {
  local models_dir="$1" split="$2" manifest="$3"
  MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$(pwd -W)/${RAW_COLOR#./}:/data:ro" \
    -v "$(pwd -W)/../docs/experiment_protocol/datasets/pv19-capped-primary-v1:/protocol:ro" \
    -v "$(pwd -W)/${models_dir#./}:/out" \
    infra_fl-client-0 python -m app.evaluate_global \
      --data-root /data \
      --manifest "/protocol/${manifest}" \
      --checkpoint /out/global_latest.npz \
      --output-json "/out/${split}_evaluation.json" \
      --img-size 128 --batch-size 32 --num-workers 0
}

verify_attack() {
  python - "$1" "$2" "$3" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
attacker = f"client-{sys.argv[2]}"
expected = float(sys.argv[3])
fit = [item for item in payload.get("history", []) if item.get("phase") == "fit"]
if len(fit) != 10:
    raise SystemExit(f"expected 10 fit rounds, got {len(fit)}")
active = 0
for item in fit:
    clients = {c["client_id"]: c for c in item.get("client_metrics", [])}
    if attacker not in clients:
        raise SystemExit(f"{attacker} missing")
    row = clients[attacker]
    if not row.get("model_update_attack"):
        raise SystemExit(f"{attacker} model_update inactive round {item.get('round')}")
    if abs(float(row.get("update_scale", 0)) - expected) > 1e-6:
        raise SystemExit(f"scale {row.get('update_scale')} != {expected}")
    active += 1
    for client_id, other in clients.items():
        if client_id != attacker and other.get("model_update_attack"):
            raise SystemExit(f"unexpected attacker {client_id}")
print(f"[pv19-mu] attack verified attacker={attacker} active_rounds={active} scale={expected}")
PY
}

write_run_manifest() {
  python - "$1/run_manifest.json" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
history = json.loads((path.parent / "fl_history.json").read_text(encoding="utf-8"))
payload = {
    "schema_version": 1,
    "experiment_id": sys.argv[2],
    "alpha_label": sys.argv[3],
    "seed": int(sys.argv[4]),
    "algorithm": sys.argv[5],
    "local_epochs": 1,
    "attack": "model_update",
    "update_scale": float(sys.argv[6]),
    "attacker_client": int(sys.argv[7]),
    "started_utc": sys.argv[8],
    "finished_utc": sys.argv[9],
    "dataset_id": "pv19-capped-62b5b2119fb2",
    "rounds": 10,
    "clients": 5,
    "role": "phase4-model-update",
    "scientific": True,
    "test_evaluated": False,
    "history_experiment_id": history.get("experiment_id"),
}
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

mkdir -p "$RUNS_ROOT"
jobs=()
for alpha_label in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for algo in "${ALGOS[@]}"; do
      for scale in "${SCALES[@]}"; do
        jobs+=("$(job_name "$alpha_label" "$seed" "$algo" "$scale")|$alpha_label|$seed|$algo|$scale")
      done
    done
  done
done

python - "$REGISTRY" "${jobs[@]}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
scale_value = {"minus05": -0.5, "minus1": -1.0}
rows = []
for item in sys.argv[2:]:
    name, alpha_label, seed, algo, scale = item.split("|")
    rows.append({
        "experiment_id": name,
        "alpha_label": alpha_label,
        "seed": int(seed),
        "algorithm": algo,
        "update_scale": scale_value[scale],
        "role": "phase4-model-update",
        "scientific": True,
    })
path.write_text(json.dumps({"schema_version": 1, "stage": "4.2c-mu", "jobs": rows}, indent=2), encoding="utf-8")
print(f"[pv19-mu] wrote {path} ({len(rows)} jobs)")
PY

echo "[pv19-mu] ${#jobs[@]} pre-registered model-update jobs"
for item in "${jobs[@]}"; do
  IFS='|' read -r name alpha_label seed algo scale <<<"$item"
  status="pending"
  if completed "$RUNS_ROOT/$name" && [[ -f "$RUNS_ROOT/$name/validation_evaluation.json" ]]; then
    status="complete"
  elif [[ -f "$RUNS_ROOT/$name/fl_history.json" ]]; then
    status="incomplete"
  fi
  echo "  $status  $name"
done

if [[ "$MODE" == "eval_test" ]]; then
  for item in "${jobs[@]}"; do
    IFS='|' read -r name alpha_label seed algo scale <<<"$item"
    models_dir="$RUNS_ROOT/$name"
    [[ -f "$models_dir/global_latest.npz" ]] || continue
    [[ -f "$models_dir/test_evaluation.json" ]] && continue
    echo "[pv19-mu] test-eval $name"
    evaluate_split "$models_dir" test test.json
  done
  echo "[pv19-mu] test evaluation pass complete"
  exit 0
fi

if [[ "$MODE" != "run" ]]; then
  echo "[pv19-mu] list-only. Set PV19_MU_MODE=run"
  exit 0
fi

export NUM_CLIENTS=5 FL_MIN_CLIENTS=5 FL_EVAL_CLIENTS=1 FL_NUM_ROUNDS=10
export FL_IMG_SIZE=128 FL_LOCAL_EPOCHS=1
export DATASET_DIR="$RAW_COLOR" EVAL_MANIFEST="$VAL_MANIFEST"

for item in "${jobs[@]}"; do
  IFS='|' read -r name alpha_label seed algo scale <<<"$item"
  [[ -n "$ONLY" && "$ONLY" != "$name" ]] && continue
  models_dir="$RUNS_ROOT/$name"
  if completed "$models_dir" && [[ -f "$models_dir/validation_evaluation.json" ]]; then
    echo "[pv19-mu] skip complete $name"
    continue
  fi
  attacker="$(attacker_client "$alpha_label" "$seed")"
  config="$(scale_config "$scale")"
  value="$(scale_value "$scale")"
  unset FL_ATTACK_CONFIG_CLIENT_0 FL_ATTACK_CONFIG_CLIENT_1 FL_ATTACK_CONFIG_CLIENT_2 FL_ATTACK_CONFIG_CLIENT_3 FL_ATTACK_CONFIG_CLIENT_4
  export "FL_ATTACK_CONFIG_CLIENT_${attacker}=$config"
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[pv19-mu] starting $name attacker=client-${attacker} scale=${value} at $started"
  echo "[pv19-mu] starting $name attacker=client-${attacker} scale=${value} at $started" >>"$LOG"
  export PARTITIONS_DIR="../docs/experiment_protocol/partitions/pv19-capped/${alpha_label}/seed-${seed}"
  export MODELS_DIR="$models_dir" FL_EXPERIMENT_ID="$name" FL_AGGREGATION="$algo" FL_SEED="$seed"
  export FL_PROXIMAL_MU=0.01
  [[ "$algo" == "fedavg" ]] && export FL_PROXIMAL_MU=0.0
  bash "$ROOT/run_fl_sequential.sh"
  verify_attack "$models_dir/fl_history.json" "$attacker" "$value"
  evaluate_validation "$models_dir"
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_run_manifest "$models_dir" "$name" "$alpha_label" "$seed" "$algo" "$value" "$attacker" "$started" "$finished"
  echo "[pv19-mu] done $name at $finished" | tee -a "$LOG"
done
echo "[pv19-mu] all requested jobs finished"
