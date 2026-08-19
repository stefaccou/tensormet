#!/bin/bash

# Canonical benchmark driver: runs the dataset × rank × divergence matrix for
# ANY solver and appends one session entry to benchmark.json.
#
#   benchmarking.sh                 # MU (multiplicative updates), the default
#   benchmarking.sh --solver sgd    # torch minibatch solver
#   benchmarking.sh --solver cp     # MU on the EXPERIMENTAL nonnegative CP family
#   benchmarking.sh --name "tc off" # free-text session label, shown in the notebook
#   benchmarking.sh --max-seconds N # per-run kill switch (default 3600, 0 = off)
#
# This is the ONE implementation. Everything else (2_benchmarking/*.sh for the
# local box, scripts/9_hpc/tier{1,2}/*.sh for PBS/Slurm) is a thin wrapper that
# loads its environment and then execs this file with a few BENCH_* overrides.
# Do not fork this script — add a knob instead.
#
# --- Configuration ------------------------------------------------------------
# Every setting below is overridable from the environment (BENCH_*) so wrappers
# never need to edit the body. List-valued knobs are space-separated strings:
#
#   BENCH_SOLVER=mu|sgd|cp  BENCH_N_GPUS=4      BENCH_METHOD=scSoftPlus
#   BENCH_NAME="tensor cores off"   # session label recorded in benchmark.json
#   BENCH_DATASETS="a b"    BENCH_RANKS="10 50" BENCH_DIVERGENCES="kl fr"
#   BENCH_DIM=10000         BENCH_ITERATIONS=4  BENCH_RUN_EXTRA=true
#   BENCH_JSON=/path/benchmark.json             BENCH_DRY_RUN=true
#   BENCH_MAX_SECONDS_PER_DECOMP=3600           # per-run watchdog, 0 disables
#   (sgd only) BENCH_SGD_LR, BENCH_SGD_BATCH_SIZE, BENCH_SGD_STEPS_PER_ITER,
#              BENCH_SGD_OPTIMIZER, BENCH_SGD_PARAMETRIZATION,
#              BENCH_SGD_WARM_START, BENCH_SGD_COMM_BACKEND,
#              BENCH_LOG_EVERY, BENCH_RUN_VARIANTS
#   (cp only)  BENCH_CP_INNER_ITERS, BENCH_CP_SCOOCH_KAPPA
#
# --- How the solvers differ, and why ------------------------------------------
# The SGD path rejects several MU-only knobs (guard rails in
# tucker_tensor.py, see sgd/README.md), so relative to MU the
# following flags are DROPPED, not merely changed:
#   --subsample-frac <1  : SGD is already minibatch -> use --sgd-batch-size.
#   --normalize-factors  : scaling lives in the softplus/clamp parametrization.
#   --largedim           : SGD does not use the largedim kernel family.
#   --max-nnz / svd init : unsupported (warm start instead).
#
# CP is the MU solver on a different model family (experimental/CP/README.md),
# so it keeps MU's cadence and subsampling and drops only:
#   --largedim           : CP has ONE kernel family; the flag has no meaning.
#   --normalize-factors  : the CP factor updates already normalize and absorb
#                          the column scales into λ every sweep.
# It is single-GPU and full-objective only (both rejected at fit time).
#
# CONSEQUENCE FOR COMPARABILITY: MU's core matrix subsamples NNZ (0.25 / 0.025),
# so it fits a *smaller tensor* than SGD does. Reconstruction errors across
# different subsample fractions are not on the same tensor and must not be
# diffed. The SGD matrix therefore runs at full data and keys its results with
# the "__nosub" suffix, i.e. the same key MU emits in its no-subsampling extra
# runs. Out of the box the aligned rows are the ones MU already runs full-data
# (fineweb-en, rank 10, kl+fr); to align more, add them to _extra_configs with
# subsample 1.0 and suffix "nosub". Other rows show "n/a" in the notebook.
#
# CP needs no such realignment: it runs MU's subsample fractions and emits MU's
# keys, and both families report the SAME relative metric (rel_KL = KL/Σx,
# rel_FR = ||X-X̂||/||X||), so a CP session diffs against an MU session row for
# row. What differs is the model, not the yardstick -- at equal rank CP is the
# strictly weaker family (a Tucker core has R^N free parameters where CP's λ
# has R), so expect worse rec at equal rank and read the Δ as the price of the
# smaller model, not as a regression.
#
# Step mapping: one SGD loop "iteration" = SGD_STEPS_PER_ITER optimizer steps,
# so BENCH_ITERATIONS is NOT wall-clock comparable across solvers; compare total
# runtime and the rec/sem levels reached, not per-iteration cost. MU and CP both
# run one full sweep per iteration, so their per-iteration times ARE comparable.
#
# --- Per-run watchdog ---------------------------------------------------------
# MAX_SECONDS_PER_DECOMP (default 3600) caps ONE combo's decomposition process.
# Without it a single hung or pathologically slow run stalls the whole matrix
# indefinitely. On expiry the process is killed and the driver moves to the next
# combo, recording a TIMEOUT row (a timeout is not a result -- it must never be
# diffed against a completed run).
#
# Results are appended to the SAME benchmark.json for every solver, with the
# session tagged "solver" -> benchmark_inspections.ipynb can tick an MU session
# and an SGD/CP session and read the Δrec / Δsem column per combo.
# One run per combo; CRASH recorded on any failure (e.g. OOM), TIMEOUT on expiry.

set -o pipefail

SOLVER="${BENCH_SOLVER:-mu}"
DRY_RUN="${BENCH_DRY_RUN:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver)   SOLVER="$2"; shift 2 ;;
        --n-gpus)   BENCH_N_GPUS="$2"; shift 2 ;;
        --method)   BENCH_METHOD="$2"; shift 2 ;;
        --name)     BENCH_NAME="$2"; shift 2 ;;
        --max-seconds) BENCH_MAX_SECONDS_PER_DECOMP="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        -h|--help)
            sed -n '3,10p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "ERROR: unknown argument '$1' (see --help)" >&2
            exit 2 ;;
    esac
done

if [[ "$SOLVER" != "mu" && "$SOLVER" != "sgd" && "$SOLVER" != "cp" ]]; then
    echo "ERROR: --solver must be 'mu', 'sgd' or 'cp', got '$SOLVER'" >&2
    exit 2
fi

read -r -a DATASETS    <<< "${BENCH_DATASETS:-fineweb-en fineweb_english_1B 4-gram-raw-fineweb-en_100000000}"
read -r -a RANKS       <<< "${BENCH_RANKS:-10 50 100}"
read -r -a DIVERGENCES <<< "${BENCH_DIVERGENCES:-kl fr}"
DIM="${BENCH_DIM:-10000}"
N_GPUS="${BENCH_N_GPUS:-2}"
METHOD="${BENCH_METHOD:-counting}"
# Free-text session label. Purely descriptive: it never touches run names or
# combo keys, so a labelled session still diffs against an unlabelled one.
NAME="${BENCH_NAME:-}"
RUN_EXTRA="${BENCH_RUN_EXTRA:-true}"
# Per-run wall-clock ceiling in seconds; 0 disables the watchdog entirely.
MAX_SECONDS_PER_DECOMP="${BENCH_MAX_SECONDS_PER_DECOMP:-3600}"

if ! [[ "$MAX_SECONDS_PER_DECOMP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: BENCH_MAX_SECONDS_PER_DECOMP must be a non-negative integer, got '$MAX_SECONDS_PER_DECOMP'" >&2
    exit 2
fi
if (( MAX_SECONDS_PER_DECOMP > 0 )) && ! command -v timeout >/dev/null 2>&1; then
    echo "ERROR: BENCH_MAX_SECONDS_PER_DECOMP=$MAX_SECONDS_PER_DECOMP but coreutils 'timeout' is not" >&2
    echo "       on PATH; set BENCH_MAX_SECONDS_PER_DECOMP=0 to run without the watchdog." >&2
    exit 2
fi

# --- SGD knobs for the core matrix (one config, so the matrix stays 1:1 with MU)
SGD_LR="${BENCH_SGD_LR:-1e-2}"
SGD_BATCH_SIZE="${BENCH_SGD_BATCH_SIZE:-4096}"
SGD_STEPS_PER_ITER="${BENCH_SGD_STEPS_PER_ITER:-100}"
SGD_OPTIMIZER="${BENCH_SGD_OPTIMIZER:-adam}"              # adam | sgd
SGD_PARAMETRIZATION="${BENCH_SGD_PARAMETRIZATION:-softplus}"  # softplus | clamp
SGD_WARM_START="${BENCH_SGD_WARM_START:-}"   # optional path to an MU model .pt used as init
# Cross-device collective for the sharded (N_GPUS > 1) path: auto | nccl | host.
# `auto` probes NCCL and falls back to pinned-host staging if it fails or hangs;
# `host` forces the fallback (slower, but always works — use it when a node's
# NCCL is broken and you need numbers); `nccl` refuses to fall back silently,
# which is what you want when a benchmark's timings must be NCCL timings.
SGD_COMM_BACKEND="${BENCH_SGD_COMM_BACKEND:-auto}"
# Ablation over optimizer / parametrization / lr on one small combo. These use
# suffixed keys, so they appear as extra rows (n/a against MU sessions).
RUN_VARIANTS="${BENCH_RUN_VARIANTS:-false}"

# --- CP knobs (CP-APR); defaults match the kernel defaults, i.e. plain sweeps.
CP_INNER_ITERS="${BENCH_CP_INNER_ITERS:-1}"
CP_SCOOCH_KAPPA="${BENCH_CP_SCOOCH_KAPPA:-0.0}"

if [[ "$SOLVER" == "sgd" ]]; then
    ITERATIONS="${BENCH_ITERATIONS:-20}"
    # Reconstruction-error cadence (--rec-log-every). Deliberately > 1, unlike
    # MU, because the two solvers spend their time in different places: the exact
    # error pass is O(nnz * prod(rank)) for both, but for MU that is ~1/5 of an
    # iteration's work, while for SGD it is essentially the WHOLE iteration --
    # the SGD_STEPS_PER_ITER optimizer steps only touch
    # SGD_STEPS_PER_ITER * SGD_BATCH_SIZE entries (409,600 at the defaults), so
    # at --rec-log-every 1 the benchmark would be timing full_relative_error, not
    # the solver. Keep LOG_EVERY a divisor of ITERATIONS: rec_errors only
    # receives samples at the log cadence, and the summary/notebook read
    # rec_errors[-1], so a non-divisor silently reports an intermediate iterate
    # as the final one.
    LOG_EVERY="${BENCH_LOG_EVERY:-5}"
    SEM_CHECK_EVERY="$ITERATIONS"
    RUN_PREFIX="bench_sgd_"
elif [[ "$SOLVER" == "cp" ]]; then
    # Same cadence as MU: one sweep per iteration, so the rows line up 1:1.
    ITERATIONS="${BENCH_ITERATIONS:-4}"
    LOG_EVERY="${BENCH_LOG_EVERY:-1}"
    SEM_CHECK_EVERY=4
    RUN_PREFIX="bench_cp_"
else
    ITERATIONS="${BENCH_ITERATIONS:-4}"
    LOG_EVERY="${BENCH_LOG_EVERY:-1}"
    SEM_CHECK_EVERY=4
    RUN_PREFIX="bench_"
fi

DATA_DIR="${SCRATCH_DATA:-$DATA}"
if [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: neither SCRATCH_DATA nor DATA is set" >&2
    exit 1
fi

BENCHMARK_JSON="${BENCH_JSON:-$DATA_DIR/benchmarking/benchmark.json}"
mkdir -p "$(dirname "$BENCHMARK_JSON")"

if [[ "$SOLVER" == "cp" ]] && (( N_GPUS > 1 )); then
    echo "ERROR: --solver cp is single-GPU only (the sharded CP path is not implemented);" >&2
    echo "       every run would raise. Set BENCH_N_GPUS=1." >&2
    exit 2
fi

if (( N_GPUS > 1 )); then
    if [[ "$SOLVER" == "sgd" ]]; then
        echo "WARNING: N_GPUS=$N_GPUS — the sharded SGD path (ShardedSGDTrainer) will be taken;" >&2
        echo "         timings are not comparable to single-GPU runs, and the effective batch" >&2
        echo "         is SGD_BATCH_SIZE per shard." >&2
    else
        echo "WARNING: N_GPUS=$N_GPUS — sharded multi-GPU path will be taken; timings are not comparable to single-GPU runs." >&2
    fi
fi

[[ -n "$NAME" ]] && echo "benchmark name:   $NAME"
echo "benchmark config: solver=$SOLVER method=$METHOD dim=$DIM n_gpus=$N_GPUS iterations=$ITERATIONS"
if (( MAX_SECONDS_PER_DECOMP > 0 )); then
    echo "                  per-run limit=${MAX_SECONDS_PER_DECOMP}s (killed and recorded TIMEOUT on expiry)"
else
    echo "                  per-run limit=off"
fi
echo "                  datasets=[${DATASETS[*]}] ranks=[${RANKS[*]}] divergences=[${DIVERGENCES[*]}]"
if [[ "$SOLVER" == "sgd" ]]; then
    echo "                  sgd=$SGD_OPTIMIZER/$SGD_PARAMETRIZATION lr=$SGD_LR bs=$SGD_BATCH_SIZE steps/iter=$SGD_STEPS_PER_ITER"
    if (( N_GPUS > 1 )); then
        echo "                  comm_backend=$SGD_COMM_BACKEND (resolved backend is logged per run)"
    fi
elif [[ "$SOLVER" == "cp" ]]; then
    echo "                  cp=EXPERIMENTAL nonnegative CP, inner_iters=$CP_INNER_ITERS scooch_kappa=$CP_SCOOCH_KAPPA"
fi
echo "                  results -> $BENCHMARK_JSON"

TMPJSON=$(mktemp "${TMPDIR:-/tmp}/bench_${SOLVER}_results_XXXXXX.json")
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

# SGD and CP artifacts carry their own stem fragment ('SGD{order}D' / 'CP{order}D'),
# but run_name is still the leading prefix, so one set of globs covers all solvers.
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

# Decomposition times are written by launch.py to a per-run *_timing.json:
# solve_seconds = summed iteration time (updates + error kernels), measured
# with device syncs on both sides of each iteration, so it is the real cost of
# decomposing; decomp_seconds = whole loop (adds in-loop semantic evaluation).
# Both exclude data loading, imports and process startup -- the overhead that
# varies per machine/partition and vanishes in long runs. The summary reports
# solve_seconds as "decomp" for exactly that reason. The fitness.json holds
# semantic-fitness dicts, no timing.
decomp_time_ms = None
solve_time_ms = None
iter_seconds = []
timing_error = None
if not timing_files:
    timing_error = f"no {run_name}_*_timing.json in {decomp_dir}"
else:
    try:
        with open(timing_files[0]) as f:
            timing = json.load(f)
        decomp_seconds = timing.get('decomp_seconds')
        if decomp_seconds is not None:
            decomp_time_ms = int(float(decomp_seconds) * 1000)
        solve_seconds = timing.get('solve_seconds')
        if solve_seconds is not None:
            solve_time_ms = int(float(solve_seconds) * 1000)
        iter_seconds = timing.get('iter_seconds') or []
        if solve_time_ms is None:
            timing_error = f"{timing_files[0]} has no solve_seconds (stale tensormet install?)"
    except Exception as exc:
        timing_error = f"could not read {timing_files[0]}: {exc}"

with open(tmpjson) as f:
    results = json.load(f)

results[key] = {
    "runtime_ms": int(runtime_ms),
    "decomp_time_ms": decomp_time_ms, # Handled separate category
    "solve_time_ms": solve_time_ms,
    "iter_seconds": iter_seconds,
    "timing_error": timing_error,
    "rec_errors": rec_errors,
    "fitness": fitness
}

with open(tmpjson, "w") as f:
    json.dump(results, f)

def _fmt(ms):
    if ms is None:
        return "N/A"
    s, rem = divmod(int(ms), 1000)
    return f"{s}s {rem:03d}ms"

if timing_error:
    print(f"!!! [{key}] decomposition time unavailable: {timing_error}", file=sys.stderr)
else:
    # Iteration 0 absorbs one-time warmup (kernel compilation, memory-pool
    # growth, first-touch allocations), so it is usually much larger than the
    # steady-state ones. Show it separately: at low --iterations it otherwise
    # dominates solve_seconds and makes the run look slower than it scales.
    warmup = ""
    if len(iter_seconds) > 1:
        steady = iter_seconds[1:]
        warmup = (f" [first iteration {iter_seconds[0]:.2f}s incl. warmup, "
                  f"then {sum(steady) / len(steady):.2f}s/iter]")
    print(f"=== [{key}] decomposition time: {_fmt(solve_time_ms)} "
          f"(sum of {len(iter_seconds)} iteration(s)), "
          f"{_fmt(decomp_time_ms)} for the loop including in-loop evaluation"
          f"{warmup}")
PYEOF
}

# _record_crash <key> [status]   status: CRASH (default) | TIMEOUT
# Both are bare sentinel strings; benchmark_inspections.ipynb knows both.
_record_crash() {
    local key="$1"
    python3 -c "
import json
with open('$TMPJSON') as f:
    r = json.load(f)
r['$key'] = '${2:-CRASH}'
with open('$TMPJSON', 'w') as f:
    json.dump(r, f)
"
}

# Per-dataset structural settings. SUBSAMPLE applies to MU and CP (SGD always
# runs full data — see the comparability note in the header).
_dataset_shape() {
    case "$1" in
        fineweb-en)
            ORDER=3; SHARED_FACTORS="1-2"; SUBSAMPLE=0.25 ;;
        4-gram-raw-fineweb-en_100000000)
            ORDER=4; SHARED_FACTORS="0-1,1-2,2-3"; SUBSAMPLE=0.025 ;;
        *)
            ORDER=3; SHARED_FACTORS="1-2"; SUBSAMPLE=0.025 ;;
    esac
}

# _build_args <dataset> <rank> <div> <subsample> <optimizer> <param> <lr> <batch>
# Fills the global NNT_ARGS array: shared flags first, then the solver-specific
# tail. Only the tail differs between MU and SGD.
_build_args() {
    local dataset="$1" rank="$2" div="$3" subsample="$4"
    local optimizer="$5" parametrization="$6" lr="$7" batch="$8"

    NNT_ARGS=(
        --dataset "$dataset"
        --method "$METHOD"
        --divergence "$div"
        --name "$RUN_NAME"
        --dim "$DIM"
        --order "$ORDER"
        --rank "$rank"
        --verbose t
        --iterations "$ITERATIONS"
        --checkpoint-saving-steps 0
        --max-cpu-frac 0.8
        --sem-error-type all
        --return-errors full
        --rec-log-every "$LOG_EVERY"
        --sem-check-every "$SEM_CHECK_EVERY"
        --shared-factors "$SHARED_FACTORS"
        --sem-primary-key simlex_all_rho
        --n-gpus "$N_GPUS"
        --random-state 1
    )

    if [[ "$SOLVER" == "sgd" ]]; then
        NNT_ARGS+=(
            --solver sgd
            --objective full
            --init random
            --sgd-lr "$lr"
            --sgd-batch-size "$batch"
            --sgd-optimizer "$optimizer"
            --sgd-parametrization "$parametrization"
            --sgd-steps-per-iteration "$SGD_STEPS_PER_ITER"
            --sgd-comm-backend "$SGD_COMM_BACKEND"
        )
        [[ -n "$SGD_WARM_START" ]] && NNT_ARGS+=(--sgd-warm-start "$SGD_WARM_START")
    elif [[ "$SOLVER" == "cp" ]]; then
        # No --largedim / --normalize-factors: see the header note.
        NNT_ARGS+=(
            --decomposition cp
            --objective full
            --subsample-frac "$subsample"
            --cp-inner-iters "$CP_INNER_ITERS"
            --cp-scooch-kappa "$CP_SCOOCH_KAPPA"
        )
    else
        NNT_ARGS+=(
            --largedim true
            --normalize-factors t
            --subsample-frac "$subsample"
        )
    fi
}

# _run_one <key> <dataset> <rank> <div> [subsample] [optimizer] [param] [lr] [batch]
# Trailing SGD knobs default to the script-level config; the ablation loop
# overrides them per variant.
_run_one() {
    local key="$1" dataset="$2" rank="$3" div="$4"
    local subsample_override="${5:-}"
    local optimizer="${6:-$SGD_OPTIMIZER}" parametrization="${7:-$SGD_PARAMETRIZATION}"
    local lr="${8:-$SGD_LR}" batch="${9:-$SGD_BATCH_SIZE}"
    local decomp_dir="$DATA_DIR/tensors/$dataset/decomposition"

    _dataset_shape "$dataset"
    [[ -n "$subsample_override" ]] && SUBSAMPLE="$subsample_override"
    RUN_NAME="${RUN_PREFIX}${key}"

    _build_args "$dataset" "$rank" "$div" "$SUBSAMPLE" \
        "$optimizer" "$parametrization" "$lr" "$batch"

    if [[ "$SOLVER" == "sgd" ]]; then
        echo -e "\n\n\n>>> [$key] (solver=sgd opt=$optimizer param=$parametrization lr=$lr bs=$batch)"
    elif [[ "$SOLVER" == "cp" ]]; then
        echo -e "\n\n\n>>> [$key] (solver=cp subsample=$SUBSAMPLE inner_iters=$CP_INNER_ITERS)"
    else
        echo -e "\n\n\n>>> [$key] (solver=mu subsample=$SUBSAMPLE)"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN: python3 -m tensormet.scripts.nnt ${NNT_ARGS[*]}"
        return
    fi

    local start end elapsed_ms exit_code
    start=$(date +%s%N)

    if (( MAX_SECONDS_PER_DECOMP > 0 )); then
        timeout -s KILL "$MAX_SECONDS_PER_DECOMP" python3 -m tensormet.scripts.nnt "${NNT_ARGS[@]}"
    else
        python3 -m tensormet.scripts.nnt "${NNT_ARGS[@]}"
    fi
    exit_code=$?

    end=$(date +%s%N)
    elapsed_ms=$(( (end - start) / 1000000 ))

    # 137 = 128 + SIGKILL, what `timeout -s KILL` exits with on expiry.
    if [[ $exit_code -eq 137 ]] && (( MAX_SECONDS_PER_DECOMP > 0 )); then
        echo "!!! [$key] TIMEOUT after ${MAX_SECONDS_PER_DECOMP}s; killed, moving to the next combo"
        _record_crash "$key" TIMEOUT
    elif [[ $exit_code -ne 0 ]]; then
        echo "!!! [$key] CRASHED (exit $exit_code)"
        _record_crash "$key"
    else
        _record_result "$key" "$elapsed_ms" "$decomp_dir" "$RUN_NAME"
        echo "=== [$key] total runtime: $(( elapsed_ms / 1000 ))s $(printf '%03d' $(( elapsed_ms % 1000 )))ms"
    fi

    _cleanup_run "$dataset" "$RUN_NAME"
}

# --- Core Matrix Runs ---
# MU subsamples per dataset; SGD runs full data and keys with "__nosub".
for dataset in "${DATASETS[@]}"; do
    for rank in "${RANKS[@]}"; do
        for div in "${DIVERGENCES[@]}"; do
            if [[ "$SOLVER" == "sgd" ]]; then
                _run_one "${dataset}__rank${rank}__${div}__nosub" "$dataset" "$rank" "$div"
            else
                _run_one "${dataset}__rank${rank}__${div}" "$dataset" "$rank" "$div"
            fi
        done
    done
done


# --- MU/CP: extra no-subsampling runs (the rows SGD can be diffed against) ----
# dataset|rank|div|subsample|suffix
_extra_configs=(
    "fineweb-en|10|kl|1.0|nosub"
    "fineweb-en|10|fr|1.0|nosub"
)

if [[ "$SOLVER" != "sgd" && "$RUN_EXTRA" == "true" ]]; then
    for config in "${_extra_configs[@]}"; do
        IFS='|' read -r dataset rank div subsample suffix <<< "$config"
        _run_one "${dataset}__rank${rank}__${div}__${suffix}" \
            "$dataset" "$rank" "$div" "$subsample"
    done
fi


# --- SGD: optimizer / parametrization / lr ablation (SGD-only rows) ----------
# dataset|rank|div|optimizer|parametrization|lr|batch|suffix
_variant_configs=(
    "fineweb-en|10|kl|sgd|softplus|1e-2|4096|plainsgd"
    "fineweb-en|10|kl|adam|clamp|1e-2|4096|clamp"
    "fineweb-en|10|kl|adam|softplus|1e-3|4096|lr1e-3"
    "fineweb-en|10|kl|adam|softplus|1e-2|16384|bs16k"
)

if [[ "$SOLVER" == "sgd" && "$RUN_VARIANTS" == "true" ]]; then
    for config in "${_variant_configs[@]}"; do
        IFS='|' read -r dataset rank div optimizer parametrization lr batch suffix <<< "$config"
        _run_one "${dataset}__rank${rank}__${div}__nosub__${suffix}" \
            "$dataset" "$rank" "$div" "" \
            "$optimizer" "$parametrization" "$lr" "$batch"
    done
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "\nDRY RUN: nothing executed, benchmark.json untouched."
    exit 0
fi


# Print summary
echo ""
echo "===== BENCHMARK SUMMARY (solver=$SOLVER) ====="
python3 - "$TMPJSON" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as f:
    results = json.load(f)


def fmt(v):
    if v is None:
        return 'N/A'
    a, b = divmod(v, 1000)
    return f'{a}s {b:03d}ms'


# Columns: total  = wall clock of the whole process (imports, data loading,
#                   sparse conversion, decomposition, saving).
#          decomp = solve_seconds, the summed per-iteration time. This is the
#                   number to compare across machines: it excludes the startup
#                   and loading overhead, which varies per node/partition and
#                   is negligible in long runs.
#          loop   = decomp_seconds, decomp plus the in-loop semantic eval.
missing_timing = []
for key, val in results.items():
    if val in ('CRASH', 'TIMEOUT'):
        print(f'  {key:<65} {val}')
    else:
        final_rec = val['rec_errors'][-1] if val['rec_errors'] else float('nan')
        final_sem = val['fitness'][-1].get('simlex_all_rho', float('nan')) if val['fitness'] else float('nan')

        total_str = fmt(val['runtime_ms'])
        decomp_str = fmt(val.get('solve_time_ms'))
        loop_str = fmt(val.get('decomp_time_ms'))
        if val.get('timing_error'):
            missing_timing.append((key, val['timing_error']))

        print(f'  {key:<65} total={total_str:<12}  decomp={decomp_str:<12}  loop={loop_str:<12}  rec={final_rec:.4f}  sem={final_sem:.4f}')

if missing_timing:
    print('')
    print('  timing unavailable (decomp/loop show N/A) for:')
    for key, err in missing_timing:
        print(f'    {key}: {err}')
PYEOF

# Append results to benchmark.json (one file for both solvers, tagged by session)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
SGD_META="$(printf '%s|%s|%s|%s|%s' "$SGD_OPTIMIZER" "$SGD_PARAMETRIZATION" "$SGD_LR" "$SGD_BATCH_SIZE" "$SGD_STEPS_PER_ITER")"
CP_META="$(printf '%s|%s' "$CP_INNER_ITERS" "$CP_SCOOCH_KAPPA")"
python3 - "$TIMESTAMP" "$N_GPUS" "$TMPJSON" "$BENCHMARK_JSON" "$SOLVER" "$METHOD" "$SGD_META" "$NAME" "$CP_META" <<'PYEOF'
import json, sys
from pathlib import Path

timestamp, n_gpus, tmpjson, benchmark_json, solver, method, sgd_meta, name, cp_meta = sys.argv[1:]

with open(tmpjson) as f:
    results = json.load(f)

# Metadata keys must stay scalars/strings; benchmark_inspections.ipynb treats any
# top-level value that is not a per-combo dict (or 'CRASH') as session metadata.
entry = {
    "time": timestamp,
    "n_gpus": int(n_gpus),
    "solver": solver,
    "method": method,
}
if name:
    entry["name"] = name
if solver == "sgd":
    optimizer, parametrization, lr, batch_size, steps = sgd_meta.split("|")
    entry["sgd_config"] = f"{optimizer}/{parametrization} lr={lr} bs={batch_size} steps/iter={steps}"
elif solver == "cp":
    inner_iters, scooch_kappa = cp_meta.split("|")
    entry["cp_config"] = f"inner_iters={inner_iters} scooch_kappa={scooch_kappa}"
entry.update(results)

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
