#!/usr/bin/env bash
# PV-19-capped stage 4.2c: clean robust aggregators (no attack).
# Needed for a true utility vs robustness comparison (RQ3).
set -euo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${PV19_CLEAN_ROBUST_MODE:-list}"
ONLY="${PV19_CLEAN_ROBUST_ONLY:-}"
SEEDS=(101 211 307 401 503)
ALPHAS=(a01 a05 iid)
AGGS=(median trimmed_mean krum multikrum)

RAW_COLOR="${PV19_RAW_COLOR:-../plantvillage/raw/color}"
VAL_MANIFEST="../docs/experiment_protocol/datasets/pv19-capped-primary-v1/validation.json"
RUNS_ROOT="../docs/experiment_protocol/runs/pv19-capped-clean-robust"
REGISTRY="$RUNS_ROOT/clean_robust_jobs.json"
LOG="$RUNS_ROOT/phase4_clean_robust.log"

job_name() {
  echo "pv19-capped-${1}-s${2}-${3}-e1-clean"
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
      --img-size 128 --batch-size 32 --num-workers 0
}

write_run_manifest() {
  python - "$1/run_manifest.json" "$2" "$3" "$4" "$5" "$6" "$7" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
history = json.loads((path.parent / "fl_history.json").read_text(encoding="utf-8"))
agg = sys.argv[5]
payload = {
    "schema_version": 1,
    "experiment_id": sys.argv[2],
    "alpha_label": sys.argv[3],
    "seed": int(sys.argv[4]),
    "algorithm": agg,
    "local_epochs": 1,
    "attack": "none",
    "started_utc": sys.argv[6],
    "finished_utc": sys.argv[7],
    "dataset_id": "pv19-capped-62b5b2119fb2",
    "rounds": 10,
    "clients": 5,
    "krum_clients_to_keep": 4 if agg == "multikrum" else (0 if agg == "krum" else None),
    "role": "phase4-clean-robust",
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
    for agg in "${AGGS[@]}"; do
      jobs+=("$(job_name "$alpha_label" "$seed" "$agg")|$alpha_label|$seed|$agg")
    done
  done
done

python - "$REGISTRY" "${jobs[@]}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = []
for item in sys.argv[2:]:
    name, alpha_label, seed, agg = item.split("|")
    rows.append({
        "experiment_id": name,
        "alpha_label": alpha_label,
        "seed": int(seed),
        "algorithm": agg,
        "role": "phase4-clean-robust",
        "scientific": True,
    })
path.write_text(json.dumps({"schema_version": 1, "stage": "4.2c-clean-robust", "jobs": rows}, indent=2), encoding="utf-8")
print(f"[pv19-clean-robust] wrote {path} ({len(rows)} jobs)")
PY

echo "[pv19-clean-robust] ${#jobs[@]} pre-registered clean robust jobs"
for item in "${jobs[@]}"; do
  IFS='|' read -r name alpha_label seed agg <<<"$item"
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
    IFS='|' read -r name alpha_label seed agg <<<"$item"
    models_dir="$RUNS_ROOT/$name"
    [[ -f "$models_dir/global_latest.npz" ]] || continue
    [[ -f "$models_dir/test_evaluation.json" ]] && continue
    echo "[pv19-clean-robust] test-eval $name"
    evaluate_split "$models_dir" test test.json
  done
  echo "[pv19-clean-robust] test evaluation pass complete"
  exit 0
fi

if [[ "$MODE" != "run" ]]; then
  echo "[pv19-clean-robust] list-only. Set PV19_CLEAN_ROBUST_MODE=run"
  exit 0
fi

export NUM_CLIENTS=5 FL_MIN_CLIENTS=5 FL_EVAL_CLIENTS=1 FL_NUM_ROUNDS=10
export FL_IMG_SIZE=128 FL_LOCAL_EPOCHS=1 FL_PROXIMAL_MU=0.0
export FL_TRIM_BETA=0.25 FL_KRUM_MALICIOUS_CLIENTS=1
export DATASET_DIR="$RAW_COLOR" EVAL_MANIFEST="$VAL_MANIFEST"
unset FL_ATTACK_CONFIG_CLIENT_0 FL_ATTACK_CONFIG_CLIENT_1 FL_ATTACK_CONFIG_CLIENT_2 FL_ATTACK_CONFIG_CLIENT_3 FL_ATTACK_CONFIG_CLIENT_4

for item in "${jobs[@]}"; do
  IFS='|' read -r name alpha_label seed agg <<<"$item"
  [[ -n "$ONLY" && "$ONLY" != "$name" ]] && continue
  models_dir="$RUNS_ROOT/$name"
  if completed "$models_dir" && [[ -f "$models_dir/validation_evaluation.json" ]]; then
    echo "[pv19-clean-robust] skip complete $name"
    continue
  fi
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[pv19-clean-robust] starting $name at $started"
  echo "[pv19-clean-robust] starting $name at $started" >>"$LOG"
  export PARTITIONS_DIR="../docs/experiment_protocol/partitions/pv19-capped/${alpha_label}/seed-${seed}"
  export MODELS_DIR="$models_dir" FL_EXPERIMENT_ID="$name" FL_SEED="$seed"
  if [[ "$agg" == "multikrum" ]]; then
    export FL_AGGREGATION=krum FL_KRUM_CLIENTS_TO_KEEP=4
  elif [[ "$agg" == "krum" ]]; then
    export FL_AGGREGATION=krum FL_KRUM_CLIENTS_TO_KEEP=0
  else
    export FL_AGGREGATION="$agg" FL_KRUM_CLIENTS_TO_KEEP=0
  fi
  bash "$ROOT/run_fl_sequential.sh"
  evaluate_split "$models_dir" validation validation.json
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_run_manifest "$models_dir" "$name" "$alpha_label" "$seed" "$agg" "$started" "$finished"
  echo "[pv19-clean-robust] done $name at $finished" | tee -a "$LOG"
done
echo "[pv19-clean-robust] all requested jobs finished"
