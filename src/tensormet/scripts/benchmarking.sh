#!/bin/bash

# Benchmark runtime across dataset × rank × divergence combinations.
# One run per combo; CRASH recorded on any failure (e.g. OOM).
# Results written to benchmark.json with a timestamp key.

DATASETS=("fineweb-en" "fineweb_english_1B" "4-gram-raw-fineweb-en_100000000")
RANKS=(10 50 100)
DIVERGENCES=("kl" "fr")
DIM=10000
N_GPUS="2"

DATA_DIR="${SCRATCH_DATA:-$DATA}"
if [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: neither SCRATCH_DATA nor DATA is set" >&2
    exit 1
fi

BENCHMARK_JSON="$DATA_DIR/benchmarking/benchmark.json"
mkdir -p "$DATA_DIR/benchmarking"

if (( N_GPUS > 1 )); then
    echo "WARNING: N_GPUS=$N_GPUS — sharded multi-GPU path will be taken; timings are not comparable to single-GPU runs." >&2
fi

TMPJSON=$(mktemp /tmp/bench_results_XXXXXX.json)
echo '{}' > "$TMPJSON"
trap "rm -f $TMPJSON" EXIT

_cleanup_run() {
    local dataset="$1"
    local run_name="$2"
    local decomp_dir="$DATA_DIR/tensors/$dataset/decomposition"
    find "$decomp_dir" -maxdepth 1 -name "${run_name}_*" -delete 2>/dev/null
    rm -rf "$decomp_dir/${run_name}_checkpoints" 2>/dev/null
}

_record_result() {
    local key="$1"
    local runtime_ms="$2"
    local decomp_dir="$3"
    local run_name="$4"
    python3 - "$key" "$runtime_ms" "$decomp_dir" "$run_name" "$TMPJSON" <<'PYEOF'
import sys, json, glob
import numpy as np

key, runtime_ms, decomp_dir, run_name, tmpjson = sys.argv[1:]

errors_files = glob.glob(f"{decomp_dir}/{run_name}_*_errors.npy")
fitness_files = glob.glob(f"{decomp_dir}/{run_name}_*_fitness.json")
timing_files = glob.glob(f"{decomp_dir}/{run_name}_*_timing.json")

rec_errors = []
if errors_files:
    try:
        rec_errors = np.load(errors_files[0]).tolist()
    except Exception:
        pass

fitness = []
if fitness_files:
    try:
        with open(fitness_files[0]) as f:
            fitness = json.load(f)
    except Exception:
        pass

# Decomposition time is written by launch.py to a per-run *_timing.json
# (decomp_seconds = decomposition loop only, excluding data loading). The
# fitness.json holds semantic-fitness dicts and carries no timing.
decomp_time_ms = None
if timing_files:
    try:
        with open(timing_files[0]) as f:
            timing = json.load(f)
        decomp_seconds = timing.get('decomp_seconds')
        if decomp_seconds is not None:
            decomp_time_ms = int(float(decomp_seconds) * 1000)
    except Exception:
        pass

with open(tmpjson) as f:
    results = json.load(f)

results[key] = {
    "runtime_ms": int(runtime_ms),
    "decomp_time_ms": decomp_time_ms, # Handled separate category
    "rec_errors": rec_errors,
    "fitness": fitness
}

with open(tmpjson, "w") as f:
    json.dump(results, f)
PYEOF
}

_record_crash() {
    local key="$1"
    python3 -c "
import json
with open('$TMPJSON') as f:
    r = json.load(f)
r['$key'] = 'CRASH'
with open('$TMPJSON', 'w') as f:
    json.dump(r, f)
"
}

# --- Core Matrix Runs ---
for dataset in "${DATASETS[@]}"; do
    if [[ "$dataset" == "fineweb-en" ]]; then
        SUBSAMPLE=0.25
        ORDER=3
        SHARED_FACTORS="1-2"
    elif [[ "$dataset" == "4-gram-raw-fineweb-en_100000000" ]]; then
        SUBSAMPLE=0.025
        ORDER=4
        SHARED_FACTORS="0-1,1-2,2-3"
    else
        SUBSAMPLE=0.025
        ORDER=3
        SHARED_FACTORS="1-2"
    fi

    for rank in "${RANKS[@]}"; do
        for div in "${DIVERGENCES[@]}"; do
            key="${dataset}__rank${rank}__${div}"
            run_name="bench_${key}"
            decomp_dir="$DATA_DIR/tensors/$dataset/decomposition"

            echo -e "\n\n\n>>> [$key]"
            start=$(date +%s%N)

            python3 -m tensormet.scripts.nnt \
                --dataset "$dataset" \
                --method counting \
                --divergence "$div" \
                --name "$run_name" \
                --dim "$DIM" \
                --order "$ORDER" \
                --rank "$rank" \
                --largedim true \
                --verbose t \
                --subsample-frac "$SUBSAMPLE" \
                --iterations 4 \
                --checkpoint-saving-steps 0 \
                --max-cpu-frac 0.8 \
                --sem-error-type all \
                --return-errors full \
                --rec-log-every 1 \
                --sem-check-every 4 \
                --normalize-factors t \
                --shared-factors "$SHARED_FACTORS" \
                --sem-primary-key simlex_all_rho \
                --n-gpus "$N_GPUS" \
                --random-state 1
            exit_code=$?

            end=$(date +%s%N)
            elapsed_ms=$(( (end - start) / 1000000 ))

            if [[ $exit_code -ne 0 ]]; then
                echo "!!! [$key] CRASHED (exit $exit_code)"
                _record_crash "$key"
            else
                _record_result "$key" "$elapsed_ms" "$decomp_dir" "$run_name"
                echo "=== [$key] total runtime: $(( elapsed_ms / 1000 ))s $(printf '%03d' $(( elapsed_ms % 1000 )))ms"
            fi

            _cleanup_run "$dataset" "$run_name"
        done
    done
done


# --- Extra Custom Runs (No-Subsampling) ---
_extra_configs=(
    "fineweb-en|10|kl|1.0|nosub"
    "fineweb-en|10|fr|1.0|nosub"
)

for config in "${_extra_configs[@]}"; do
    IFS='|' read -r dataset rank div subsample suffix <<< "$config"

    if [[ "$dataset" == "fineweb-en" ]]; then
        ORDER=3; SHARED_FACTORS="1-2"
    elif [[ "$dataset" == "4-gram-raw-fineweb-en_100000000" ]]; then
        ORDER=4; SHARED_FACTORS="0-1,1-2,2-3"
    else
        ORDER=3; SHARED_FACTORS="1-2"
    fi

    key="${dataset}__rank${rank}__${div}__${suffix}"
    run_name="bench_${key}"
    decomp_dir="$DATA_DIR/tensors/$dataset/decomposition"

    echo -e "\n\n\n>>> [$key] (Custom Run)"
    start=$(date +%s%N)

    python3 -m tensormet.scripts.nnt \
        --dataset "$dataset" \
        --method counting \
        --divergence "$div" \
        --name "$run_name" \
        --dim "$DIM" \
        --order "$ORDER" \
        --rank "$rank" \
        --largedim true \
        --verbose t \
        --subsample-frac "$subsample" \
        --iterations 4 \
        --checkpoint-saving-steps 0 \
        --max-cpu-frac 0.8 \
        --sem-error-type all \
        --return-errors full \
        --rec-log-every 1 \
        --sem-check-every 4 \
        --normalize-factors t \
        --shared-factors "$SHARED_FACTORS" \
        --sem-primary-key simlex_all_rho \
        --n-gpus "$N_GPUS" \
        --random-state 1
    exit_code=$?

    end=$(date +%s%N)
    elapsed_ms=$(( (end - start) / 1000000 ))

    if [[ $exit_code -ne 0 ]]; then
        echo "!!! [$key] CRASHED (exit $exit_code)"
        _record_crash "$key"
    else
        _record_result "$key" "$elapsed_ms" "$decomp_dir" "$run_name"
        echo "=== [$key] total runtime: $(( elapsed_ms / 1000 ))s $(printf '%03d' $(( elapsed_ms % 1000 )))ms"
    fi

    _cleanup_run "$dataset" "$run_name"
done


# Print summary
echo ""
echo "===== BENCHMARK SUMMARY ====="
python3 -c "
import json
with open('$TMPJSON') as f:
    results = json.load(f)
for key, val in results.items():
    if val == 'CRASH':
        print(f'  {key:<65} CRASH')
    else:
        final_rec = val['rec_errors'][-1] if val['rec_errors'] else float('nan')
        final_sem = val['fitness'][-1].get('simlex_all_rho', float('nan')) if val['fitness'] else float('nan')

        s, ms = divmod(val['runtime_ms'], 1000)

        if val.get('decomp_time_ms') is not None:
            ds, dms = divmod(val['decomp_time_ms'], 1000)
            decomp_str = f'{ds}s {dms:03d}ms'
        else:
            decomp_str = 'N/A'

        print(f'  {key:<65} total={s}s {ms:03d}ms  decomp={decomp_str:<10}  rec={final_rec:.4f}  sem={final_sem:.4f}')
"

# Append results to benchmark.json
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
python3 - "$TIMESTAMP" "$N_GPUS" "$TMPJSON" "$BENCHMARK_JSON" <<'PYEOF'
import json, sys
from pathlib import Path

timestamp, n_gpus, tmpjson, benchmark_json = sys.argv[1:]

with open(tmpjson) as f:
    results = json.load(f)

entry = {"time": timestamp, "n_gpus": int(n_gpus), **results}

if Path(benchmark_json).exists():
    with open(benchmark_json) as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
else:
    data = []

data.append(entry)
with open(benchmark_json, "w") as f:
    json.dump(data, f, indent=2)
print(f"Saved results to {benchmark_json}")
PYEOF