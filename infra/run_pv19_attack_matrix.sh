#!/usr/bin/env bash
# PV-19-capped phase 4.1 primary label-flip matrix.
#
# Locked pairs, flip fraction 1.0, FedAvg/FedProx, E=1, all alphas and seeds.
# Defense aggregators, other fractions, and model-update are stage 4.2+.
#
#   bash run_pv19_attack_matrix.sh
#   PV19_ATTACK_MODE=run bash run_pv19_attack_matrix.sh
#   PV19_ATTACK_MODE=run PV19_ATTACK_ONLY=pv19-capped-a05-s101-fedavg-e1-flip1-apple \
#     bash run_pv19_attack_matrix.sh
#   PV19_ATTACK_MODE=eval_test bash run_pv19_attack_matrix.sh
set -euo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${PV19_ATTACK_MODE:-list}"
ONLY="${PV19_ATTACK_ONLY:-}"
SEEDS=(101 211 307 401 503)
ALPHAS=(a01 a05 iid)
ALGOS=(fedavg fedprox)
PAIRS=(apple cherry potato)

RAW_COLOR="../iot-edge/sensors/data/PlantVillage/PlantVillage-Dataset-master/PlantVillage-Dataset-master/raw/color"
VAL_MANIFEST="../docs/experiment_protocol/datasets/pv19-capped-primary-v1/validation.json"
TEST_MANIFEST="../docs/experiment_protocol/datasets/pv19-capped-primary-v1/test.json"
RUNS_ROOT="../docs/experiment_protocol/runs/pv19-capped-attack"
ATTACKERS="../docs/experiment_protocol/attack_attacker_clients.json"
REGISTRY="$RUNS_ROOT/attack_matrix_jobs.json"
LOG="$RUNS_ROOT/phase4_matrix.log"

pair_config() {
  case "$1" in
    apple) echo "/attack-configs/label_flip_apple_healthy_scab.json" ;;
    cherry) echo "/attack-configs/label_flip_cherry_healthy_mildew.json" ;;
    potato) echo "/attack-configs/label_flip_potato_healthy_late_blight.json" ;;
    *) echo "unknown" ;;
  esac
}

pair_source() {
  case "$1" in
    apple) echo "Apple___healthy" ;;
    cherry) echo "Cherry___healthy" ;;
    potato) echo "Potato___healthy" ;;
    *) echo "unknown" ;;
  esac
}

job_name() {
  local alpha_label="$1" seed="$2" algo="$3" pair="$4"
  echo "pv19-capped-${alpha_label}-s${seed}-${algo}-e1-flip1-${pair}"
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
  local pair="$1" alpha_label="$2" seed="$3"
  python - "$ATTACKERS" "$pair" "$alpha_label" "$seed" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
slug = sys.argv[2]
alpha = sys.argv[3]
seed = sys.argv[4]
source_by_slug = {
    "apple": "Apple___healthy",
    "cherry": "Cherry___healthy",
    "potato": "Potato___healthy",
}
source = source_by_slug[slug]
for pair in payload["pairs"]:
    if pair["source"] == source:
        print(pair["attackers"][alpha][seed])
        break
else:
    raise SystemExit(f"attacker not found for {slug} {alpha} {seed}")
PY
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

evaluate_validation() {
  evaluate_split "$1" validation validation.json
}

mark_test_evaluated() {
  local models_dir="$1"
  python - "$models_dir/run_manifest.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
payload["test_evaluated"] = True
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

verify_attack() {
  local history="$1" attacker="$2" source_class="$3"
  python - "$history" "$attacker" "$source_class" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
attacker = f"client-{sys.argv[2]}"
source = sys.argv[3]
fit = [item for item in payload.get("history", []) if item.get("phase") == "fit"]
if len(fit) != 10:
    raise SystemExit(f"expected 10 fit rounds, got {len(fit)}")
active = 0
flipped = 0
for item in fit:
    clients = {c["client_id"]: c for c in item.get("client_metrics", [])}
    if attacker not in clients:
        raise SystemExit(f"{attacker} missing from round {item.get('round')}")
    row = clients[attacker]
    if not row.get("label_flip_attack"):
        raise SystemExit(f"{attacker} label_flip inactive in round {item.get('round')}")
    if int(row.get("label_flip_source_count", 0)) <= 0:
        raise SystemExit(f"{attacker} has no {source} samples to flip")
    active += 1
    flipped += int(row.get("label_flipped_samples", 0))
    for client_id, other in clients.items():
        if client_id != attacker and other.get("label_flip_attack"):
            raise SystemExit(f"unexpected attacker {client_id}")
print(f"[pv19-attack] attack verified attacker={attacker} active_rounds={active} flipped_total={flipped}")
PY
}

write_run_manifest() {
  local models_dir="$1" name="$2" alpha_label="$3" seed="$4" algo="$5" pair="$6" attacker="$7" started="$8" finished="$9"
  python - "$models_dir/run_manifest.json" "$name" "$alpha_label" "$seed" "$algo" "$pair" "$attacker" "$started" "$finished" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
history_path = path.parent / "fl_history.json"
history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else {}
payload = {
    "schema_version": 1,
    "experiment_id": sys.argv[2],
    "alpha_label": sys.argv[3],
    "seed": int(sys.argv[4]),
    "algorithm": sys.argv[5],
    "local_epochs": 1,
    "pair": sys.argv[6],
    "attacker_client": int(sys.argv[7]),
    "flip_fraction": 1.0,
    "started_utc": sys.argv[8],
    "finished_utc": sys.argv[9],
    "dataset_id": "pv19-capped-62b5b2119fb2",
    "rounds": 10,
    "clients": 5,
    "eval_clients": 1,
    "proximal_mu": 0.01 if sys.argv[5] == "fedprox" else 0.0,
    "role": "phase4-label-flip-primary",
    "scientific": True,
    "test_evaluated": False,
    "history_experiment_id": history.get("experiment_id"),
    "aggregation": history.get("aggregation"),
}
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

mkdir -p "$RUNS_ROOT"
jobs=()
for alpha_label in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for algo in "${ALGOS[@]}"; do
      for pair in "${PAIRS[@]}"; do
        jobs+=("$(job_name "$alpha_label" "$seed" "$algo" "$pair")|$alpha_label|$seed|$algo|$pair")
      done
    done
  done
done

python - "$REGISTRY" "${jobs[@]}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = []
for item in sys.argv[2:]:
    name, alpha_label, seed, algo, pair = item.split("|")
    rows.append({
        "experiment_id": name,
        "alpha_label": alpha_label,
        "seed": int(seed),
        "algorithm": algo,
        "local_epochs": 1,
        "pair": pair,
        "flip_fraction": 1.0,
        "rounds": 10,
        "clients": 5,
        "dataset": "pv19-capped-62b5b2119fb2",
        "role": "phase4-label-flip-primary",
        "scientific": True,
    })
path.write_text(json.dumps({"schema_version": 1, "stage": "4.1", "jobs": rows}, indent=2), encoding="utf-8")
print(f"[pv19-attack] wrote {path} ({len(rows)} jobs)")
PY

echo "[pv19-attack] ${#jobs[@]} pre-registered stage 4.1 jobs"
for item in "${jobs[@]}"; do
  IFS='|' read -r name alpha_label seed algo pair <<<"$item"
  models_dir="$RUNS_ROOT/$name"
  status="pending"
  if completed "$models_dir" && [[ -f "$models_dir/validation_evaluation.json" ]]; then
    status="complete"
  elif [[ -f "$models_dir/fl_history.json" ]]; then
    status="incomplete"
  fi
  echo "  $status  $name"
done

if [[ "$MODE" != "run" && "$MODE" != "eval_test" ]]; then
  echo "[pv19-attack] list-only. Set PV19_ATTACK_MODE=run to execute missing jobs."
  echo "[pv19-attack] Final test evaluation is deferred until PV19_ATTACK_MODE=eval_test."
  exit 0
fi

if [[ "$MODE" == "eval_test" ]]; then
  for item in "${jobs[@]}"; do
    IFS='|' read -r name alpha_label seed algo pair <<<"$item"
    if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
      continue
    fi
    models_dir="$RUNS_ROOT/$name"
    if [[ ! -f "$models_dir/global_latest.npz" ]]; then
      echo "[pv19-attack] skip missing checkpoint $name"
      continue
    fi
    if [[ -f "$models_dir/test_evaluation.json" ]]; then
      echo "[pv19-attack] skip existing test eval $name"
      continue
    fi
    echo "[pv19-attack] test-eval $name"
    evaluate_split "$models_dir" test test.json
    mark_test_evaluated "$models_dir"
  done
  echo "[pv19-attack] test evaluation pass complete"
  exit 0
fi

export NUM_CLIENTS=5
export FL_MIN_CLIENTS=5
export FL_EVAL_CLIENTS=1
export FL_NUM_ROUNDS=10
export FL_IMG_SIZE=128
export FL_LOCAL_EPOCHS=1
export DATASET_DIR="$RAW_COLOR"
export EVAL_MANIFEST="$VAL_MANIFEST"

for item in "${jobs[@]}"; do
  IFS='|' read -r name alpha_label seed algo pair <<<"$item"
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
    continue
  fi
  models_dir="$RUNS_ROOT/$name"
  if completed "$models_dir" && [[ -f "$models_dir/validation_evaluation.json" ]]; then
    echo "[pv19-attack] skip complete $name"
    continue
  fi
  attacker="$(attacker_client "$pair" "$alpha_label" "$seed")"
  config="$(pair_config "$pair")"
  source_class="$(pair_source "$pair")"
  unset FL_ATTACK_CONFIG_CLIENT_0 FL_ATTACK_CONFIG_CLIENT_1 FL_ATTACK_CONFIG_CLIENT_2 FL_ATTACK_CONFIG_CLIENT_3 FL_ATTACK_CONFIG_CLIENT_4
  export "FL_ATTACK_CONFIG_CLIENT_${attacker}=$config"
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[pv19-attack] starting $name attacker=client-${attacker} at $started"
  echo "[pv19-attack] starting $name attacker=client-${attacker} at $started" >>"$LOG"
  export PARTITIONS_DIR="../docs/experiment_protocol/partitions/pv19-capped/${alpha_label}/seed-${seed}"
  export MODELS_DIR="$models_dir"
  export FL_EXPERIMENT_ID="$name"
  export FL_AGGREGATION="$algo"
  export FL_SEED="$seed"
  if [[ "$algo" == "fedprox" ]]; then
    export FL_PROXIMAL_MU=0.01
  else
    export FL_PROXIMAL_MU=0.0
  fi
  bash "$ROOT/run_fl_sequential.sh"
  verify_attack "$models_dir/fl_history.json" "$attacker" "$source_class"
  evaluate_validation "$models_dir"
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_run_manifest "$models_dir" "$name" "$alpha_label" "$seed" "$algo" "$pair" "$attacker" "$started" "$finished"
  echo "[pv19-attack] done $name at $finished"
  echo "[pv19-attack] done $name at $finished" >>"$LOG"
done
echo "[pv19-attack] all requested jobs finished"
