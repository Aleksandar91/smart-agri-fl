#!/usr/bin/env bash
# Phase 5: locked PV-19-full confirmation (45 jobs). See phase5_locked.md.
set -euo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${PV19_FULL_MODE:-list}"
ONLY="${PV19_FULL_ONLY:-}"
SEEDS=(101 211 307 401 503)

RAW_COLOR="../iot-edge/sensors/data/PlantVillage/PlantVillage-Dataset-master/PlantVillage-Dataset-master/raw/color"
VAL_MANIFEST="../docs/experiment_protocol/datasets/pv19-full-confirmatory-v1/validation.json"
RUNS_ROOT="../docs/experiment_protocol/runs/pv19-full-confirm"
PART_ROOT="../docs/experiment_protocol/partitions/pv19-full"
REGISTRY="$RUNS_ROOT/full_confirm_jobs.json"
LOG="$RUNS_ROOT/phase5_full.log"
FLIP_CFG="/attack-configs/label_flip_apple_healthy_scab.json"

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

apple_attacker() {
  python - "$PART_ROOT/$1/seed-$2/partitions_summary.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
best_id, best_n = None, -1
for client in payload["clients"]:
    n = int(client.get("per_class", {}).get("Apple___healthy", 0))
    cid = int(client["client_id"])
    if n > best_n or (n == best_n and (best_id is None or cid < best_id)):
        best_n, best_id = n, cid
if best_id is None or best_n <= 0:
    raise SystemExit("no Apple___healthy owner")
print(best_id)
PY
}

evaluate_split() {
  local models_dir="$1" split="$2" manifest="$3"
  MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$(pwd -W)/${RAW_COLOR#./}:/data:ro" \
    -v "$(pwd -W)/../docs/experiment_protocol/datasets/pv19-full-confirmatory-v1:/protocol:ro" \
    -v "$(pwd -W)/${models_dir#./}:/out" \
    infra_fl-client-0 python -m app.evaluate_global \
      --data-root /data \
      --manifest "/protocol/${manifest}" \
      --checkpoint /out/global_latest.npz \
      --output-json "/out/${split}_evaluation.json" \
      --img-size 128 --batch-size 32 --num-workers 0
}

mkdir -p "$RUNS_ROOT"
jobs=()
for seed in "${SEEDS[@]}"; do
  jobs+=("pv19-full-a01-s${seed}-fedavg-e1|a01|$seed|fedavg|clean|fedavg")
  jobs+=("pv19-full-a01-s${seed}-fedprox-e1|a01|$seed|fedprox|clean|fedprox")
  jobs+=("pv19-full-iid-s${seed}-fedavg-e1|iid|$seed|fedavg|clean|fedavg")
  jobs+=("pv19-full-a01-s${seed}-fedavg-e1-flip1-apple|a01|$seed|fedavg|flip1-apple|fedavg")
  jobs+=("pv19-full-a05-s${seed}-fedavg-e1-flip1-apple|a05|$seed|fedavg|flip1-apple|fedavg")
  jobs+=("pv19-full-iid-s${seed}-fedavg-e1-flip1-apple|iid|$seed|fedavg|flip1-apple|fedavg")
  jobs+=("pv19-full-a01-s${seed}-median-e1-flip1-apple|a01|$seed|median|flip1-apple|median")
  jobs+=("pv19-full-a05-s${seed}-median-e1-flip1-apple|a05|$seed|median|flip1-apple|median")
  jobs+=("pv19-full-iid-s${seed}-median-e1-flip1-apple|iid|$seed|median|flip1-apple|median")
done

python - "$REGISTRY" "${jobs[@]}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = []
for item in sys.argv[2:]:
    name, alpha_label, seed, algo, role, agg = item.split("|")
    rows.append({
        "experiment_id": name,
        "alpha_label": alpha_label,
        "seed": int(seed),
        "algorithm": algo,
        "role": f"phase5-{role}",
        "aggregation": agg,
        "dataset": "pv19-full-774007483a1d",
        "scientific": True,
    })
path.write_text(json.dumps({"schema_version": 1, "stage": "5", "jobs": rows}, indent=2), encoding="utf-8")
print(f"[pv19-full] wrote {path} ({len(rows)} jobs)")
PY

echo "[pv19-full] ${#jobs[@]} pre-registered confirmatory jobs"
for item in "${jobs[@]}"; do
  IFS='|' read -r name alpha_label seed algo role agg <<<"$item"
  status="pending"
  if completed "$RUNS_ROOT/$name" && [[ -f "$RUNS_ROOT/$name/validation_evaluation.json" ]]; then
    status="complete"
  fi
  echo "  $status  $name"
done

if [[ "$MODE" == "eval_test" ]]; then
  for item in "${jobs[@]}"; do
    IFS='|' read -r name alpha_label seed algo role agg <<<"$item"
    models_dir="$RUNS_ROOT/$name"
    [[ -f "$models_dir/global_latest.npz" ]] || continue
    [[ -f "$models_dir/test_evaluation.json" ]] && continue
    echo "[pv19-full] test-eval $name"
    evaluate_split "$models_dir" test test.json
  done
  echo "[pv19-full] test evaluation pass complete"
  exit 0
fi

if [[ "$MODE" != "run" ]]; then
  echo "[pv19-full] list-only. Set PV19_FULL_MODE=run"
  exit 0
fi

export NUM_CLIENTS=5 FL_MIN_CLIENTS=5 FL_EVAL_CLIENTS=1 FL_NUM_ROUNDS=10
export FL_IMG_SIZE=128 FL_LOCAL_EPOCHS=1 FL_TRIM_BETA=0.25
export DATASET_DIR="$RAW_COLOR" EVAL_MANIFEST="$VAL_MANIFEST"

for item in "${jobs[@]}"; do
  IFS='|' read -r name alpha_label seed algo role agg <<<"$item"
  [[ -n "$ONLY" && "$ONLY" != "$name" ]] && continue
  models_dir="$RUNS_ROOT/$name"
  if completed "$models_dir" && [[ -f "$models_dir/validation_evaluation.json" ]]; then
    echo "[pv19-full] skip complete $name"
    continue
  fi
  unset FL_ATTACK_CONFIG_CLIENT_0 FL_ATTACK_CONFIG_CLIENT_1 FL_ATTACK_CONFIG_CLIENT_2 FL_ATTACK_CONFIG_CLIENT_3 FL_ATTACK_CONFIG_CLIENT_4
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[pv19-full] starting $name at $started"
  echo "[pv19-full] starting $name at $started" >>"$LOG"
  export PARTITIONS_DIR="${PART_ROOT}/${alpha_label}/seed-${seed}"
  export MODELS_DIR="$models_dir" FL_EXPERIMENT_ID="$name" FL_SEED="$seed"
  export FL_PROXIMAL_MU=0.0 FL_KRUM_CLIENTS_TO_KEEP=0
  if [[ "$algo" == "fedprox" ]]; then
    export FL_AGGREGATION=fedprox FL_PROXIMAL_MU=0.01
  elif [[ "$algo" == "median" ]]; then
    export FL_AGGREGATION=median
  else
    export FL_AGGREGATION=fedavg
  fi
  if [[ "$role" == "flip1-apple" ]]; then
    attacker="$(apple_attacker "$alpha_label" "$seed")"
    export "FL_ATTACK_CONFIG_CLIENT_${attacker}=$FLIP_CFG"
    echo "[pv19-full] attacker=client-${attacker} for $name"
  fi
  bash "$ROOT/run_fl_sequential.sh"
  evaluate_split "$models_dir" validation validation.json
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[pv19-full] done $name at $finished" | tee -a "$LOG"
done
echo "[pv19-full] all requested jobs finished"
