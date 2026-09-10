#!/usr/bin/env bash
# PV-19-capped clean utility matrix (Phase 3).
#
# Default is list-only. Full execution is long (60 runs × 10 rounds).
#
#   bash run_pv19_utility_matrix.sh
#   PV19_MATRIX_MODE=run bash run_pv19_utility_matrix.sh
#   PV19_MATRIX_MODE=run PV19_MATRIX_ONLY=pv19-capped-a05-s101-fedavg-e1 \
#     bash run_pv19_utility_matrix.sh
set -euo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${PV19_MATRIX_MODE:-list}"
ONLY="${PV19_MATRIX_ONLY:-}"
SEEDS=(101 211 307 401 503)
ALPHAS=(a01 a05 iid)
ALGOS=(fedavg fedprox)
EPOCHS=(1 5)

RAW_COLOR="../iot-edge/sensors/data/PlantVillage/PlantVillage-Dataset-master/PlantVillage-Dataset-master/raw/color"
VAL_MANIFEST="../docs/experiment_protocol/datasets/pv19-capped-primary-v1/validation.json"
TEST_MANIFEST="../docs/experiment_protocol/datasets/pv19-capped-primary-v1/test.json"
RUNS_ROOT="../docs/experiment_protocol/runs/pv19-capped"
REGISTRY="$RUNS_ROOT/utility_matrix_jobs.json"

alpha_value() {
  case "$1" in
    a01) echo "0.1" ;;
    a05) echo "0.5" ;;
    iid) echo "iid" ;;
    *) echo "unknown" ;;
  esac
}

job_name() {
  local alpha_label="$1" seed="$2" algo="$3" epochs="$4"
  echo "pv19-capped-${alpha_label}-s${seed}-${algo}-e${epochs}"
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
      --img-size 128 \
      --batch-size 32 \
      --num-workers 0
}

write_run_manifest() {
  local models_dir="$1" name="$2" alpha_label="$3" seed="$4" algo="$5" epochs="$6" started="$7" finished="$8"
  python - "$models_dir/run_manifest.json" "$name" "$alpha_label" "$seed" "$algo" "$epochs" "$started" "$finished" <<'PY'
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
    "local_epochs": int(sys.argv[6]),
    "started_utc": sys.argv[7],
    "finished_utc": sys.argv[8],
    "dataset_id": "pv19-capped-62b5b2119fb2",
    "rounds": 10,
    "clients": 5,
    "eval_clients": 1,
    "proximal_mu": 0.01 if sys.argv[5] == "fedprox" else 0.0,
    "role": "clean-utility-matrix",
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
      for epochs in "${EPOCHS[@]}"; do
        jobs+=("$(job_name "$alpha_label" "$seed" "$algo" "$epochs")|$alpha_label|$seed|$algo|$epochs")
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
    name, alpha_label, seed, algo, epochs = item.split("|")
    rows.append({
        "experiment_id": name,
        "alpha_label": alpha_label,
        "seed": int(seed),
        "algorithm": algo,
        "local_epochs": int(epochs),
        "rounds": 10,
        "clients": 5,
        "eval_clients": 1,
        "dataset": "pv19-capped-62b5b2119fb2",
        "role": "clean-utility-matrix",
        "scientific": True,
    })
path.write_text(json.dumps({"schema_version": 1, "jobs": rows}, indent=2), encoding="utf-8")
print(f"[pv19-matrix] wrote {path} ({len(rows)} jobs)")
PY

echo "[pv19-matrix] ${#jobs[@]} pre-registered clean utility jobs"
for item in "${jobs[@]}"; do
  IFS='|' read -r name alpha_label seed algo epochs <<<"$item"
  models_dir="$RUNS_ROOT/$name"
  status="pending"
  if completed "$models_dir"; then
    status="complete"
  elif [[ -f "$models_dir/fl_history.json" ]]; then
    status="incomplete"
  fi
  echo "  $status  $name"
done

if [[ "$MODE" != "run" && "$MODE" != "eval_test" ]]; then
  echo "[pv19-matrix] list-only. Set PV19_MATRIX_MODE=run to execute missing jobs."
  echo "[pv19-matrix] Final test evaluation is deferred until PV19_MATRIX_MODE=eval_test."
  exit 0
fi

if [[ "$MODE" == "eval_test" ]]; then
  for item in "${jobs[@]}"; do
    IFS='|' read -r name alpha_label seed algo epochs <<<"$item"
    if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
      continue
    fi
    models_dir="$RUNS_ROOT/$name"
    if [[ ! -f "$models_dir/global_latest.npz" ]]; then
      echo "[pv19-matrix] skip missing checkpoint $name"
      continue
    fi
    if [[ -f "$models_dir/test_evaluation.json" ]]; then
      echo "[pv19-matrix] skip existing test eval $name"
      continue
    fi
    echo "[pv19-matrix] test-eval $name"
    evaluate_split "$models_dir" test test.json
  done
  echo "[pv19-matrix] test evaluation pass complete"
  exit 0
fi

export NUM_CLIENTS=5
export FL_MIN_CLIENTS=5
export FL_EVAL_CLIENTS=1
export FL_NUM_ROUNDS=10
export FL_IMG_SIZE=128
export DATASET_DIR="$RAW_COLOR"
export EVAL_MANIFEST="$VAL_MANIFEST"
LOG="$RUNS_ROOT/phase3_matrix.log"

for item in "${jobs[@]}"; do
  IFS='|' read -r name alpha_label seed algo epochs <<<"$item"
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
    continue
  fi
  models_dir="$RUNS_ROOT/$name"
  if completed "$models_dir" && [[ -f "$models_dir/validation_evaluation.json" ]]; then
    echo "[pv19-matrix] skip complete $name"
    continue
  fi
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[pv19-matrix] starting $name at $started"
  echo "[pv19-matrix] starting $name at $started" >>"$LOG"
  export PARTITIONS_DIR="../docs/experiment_protocol/partitions/pv19-capped/${alpha_label}/seed-${seed}"
  export MODELS_DIR="$models_dir"
  export FL_EXPERIMENT_ID="$name"
  export FL_AGGREGATION="$algo"
  export FL_LOCAL_EPOCHS="$epochs"
  export FL_SEED="$seed"
  if [[ "$algo" == "fedprox" ]]; then
    export FL_PROXIMAL_MU=0.01
  else
    export FL_PROXIMAL_MU=0.0
  fi
  bash "$ROOT/run_fl_sequential.sh"
  evaluate_split "$models_dir" validation validation.json
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_run_manifest "$models_dir" "$name" "$alpha_label" "$seed" "$algo" "$epochs" "$started" "$finished"
  echo "[pv19-matrix] done $name at $finished"
  echo "[pv19-matrix] done $name at $finished" >>"$LOG"
done
echo "[pv19-matrix] all requested jobs finished"
