#!/bin/bash

# Canonical benchmark driver: runs every entry of the benchmark folder
# ($DATA/tensors/benchmarking/) × rank × divergence for ANY variant and appends
# one session entry to benchmark.json.
#
#   benchmarking.sh                 # tt: MU on the Tucker-TT hybrid (the default)
#   benchmarking.sh utt             # MU on the shared-core TT
#   benchmarking.sh mu              # MU on the full Tucker core
#   benchmarking.sh sgd             # torch minibatch solver (Tucker)
#   benchmarking.sh sgd-tt          # torch minibatch solver (Tucker-TT hybrid)
#   benchmarking.sh cp              # MU on the EXPERIMENTAL nonnegative CP family
#   benchmarking.sh --solver utt    # same as the bare word
#   benchmarking.sh --entries "a b" # only these entries (default: all of them)
#   benchmarking.sh --name "tc off" # free-text session label, shown in the notebook
#   benchmarking.sh --max-seconds N # per-run kill switch (default 3600, 0 = off)
#   benchmarking.sh --dry-run       # print the commands, run nothing
#
# --- Inputs: the benchmark folder and the settings file -----------------------
# One subfolder per test tensor, made by tensormet.scripts.add_benchmark_entry:
#   <entry>/entry.conf                       what the tensor is (KEY=value: ORDER,
#                                            SHARED_FACTORS, DIM, METHOD)
#   <entry>/vocabularies/*.pkl               copied from the source dataset
#   <entry>/populated/{METHOD}_{ORDER}D_{DIM}d*.pt
#   <entry>/eval_sentences/sample.parquet    launch.py reads eval sentences here
#   <entry>/decomposition/                   benchmark runs write here, then get deleted
# What to run lives in one settings file (default 2_benchmarking/benchmark_settings.conf,
# or BENCH_SETTINGS), one line per entry:
#   <entry> <subsample> <ranks> <full_data_ranks> <variants>    (lists comma-separated, "-" = none)
# An entry without a line is not run. A variant not listed for an entry skips
# it; utt also skips entries whose modes are not all tied.
#
# --- Configuration ------------------------------------------------------------
# Every setting below is overridable from the environment (BENCH_*) so wrappers
# never need to edit the body. List-valued knobs are space-separated strings:
#
#   BENCH_SOLVER=tt|utt|mu|sgd|sgd-tt|cp  BENCH_N_GPUS=2
#   BENCH_NAME="tensor cores off"   # session label recorded in benchmark.json
#   BENCH_FOLDER=benchmarking       # folder under $DATA/tensors/
#   BENCH_SETTINGS=/path/benchmark_settings.conf
#   BENCH_ENTRIES="a b"             # default: every entry in the settings file
#   BENCH_RANKS="10 50"             # overrides the ranks of every entry
#   BENCH_FULL_DATA=true            # false skips the full_data_ranks rows
#   BENCH_DIVERGENCES="kl"  # default; "kl fr" re-enables the FR rows
#   BENCH_ITERATIONS=4
#   BENCH_INIT=random       # init for every variant except sgd (always random)
#   BENCH_JSON=/path/benchmark.json             BENCH_DRY_RUN=true
#   BENCH_MAX_SECONDS_PER_DECOMP=3600           # per-run watchdog, 0 disables
#   (sgd, sgd-tt) BENCH_SGD_LR, BENCH_SGD_BATCH_SIZE, BENCH_SGD_STEPS_PER_ITER,
#              BENCH_SGD_PARAMETRIZATION,
#              BENCH_SGD_WARM_START, BENCH_SGD_COMM_BACKEND, BENCH_LOG_EVERY
#   (cp only)  BENCH_CP_INNER_ITERS, BENCH_CP_SCOOCH_KAPPA
#   (tt, utt, sgd-tt) BENCH_TT_RANK             # bond dimension, default 100
#   (utt only) BENCH_UTT_CORE_POWER             # float or "auto"; unset = 1
#
# --- Result keys ----------------------------------------------------------------
# <entry>__rank<r>__<div>__ss<frac>. MU-family rows use the entry's subsample;
# SGD always runs the full tensor (ss1), and so do the full_data_ranks rows, so
# those are the rows where an MU-family session and an SGD session line up.
# Rows at different fractions fit different data and never share a key.
#
# Results go to $DATA/benchmarking/benchmark.json, tagged "protocol": 2. Sessions
# from the old fixed dataset list are moved once to benchmark_old.json.
#
# --- How the solvers differ, and why ------------------------------------------
# The SGD path rejects several MU-only knobs (guard rails in
# tucker_tensor.py, see sgd/README.md), so relative to MU the
# following flags are DROPPED, not merely changed:
#   --subsample-frac <1  : SGD is already minibatch -> use --sgd-batch-size.
#                          (passed as 1.0, since the config default is 0.025)
#   --normalize-factors  : scaling lives in the softplus/exp parametrization.
#   --largedim           : SGD does not use the largedim kernel family.
#   --max-nnz / svd init : unsupported (warm start instead).
#
# CP is the MU solver on a different model family (experimental/CP/README.md),
# so it keeps MU's cadence and subsampling and drops only:
#   --largedim           : CP has ONE kernel family; the flag has no meaning.
#   --normalize-factors  : the CP factor updates already normalize and absorb
#                          the column scales into λ every sweep.
# It is full-objective only (masked is rejected at fit time). Multi-GPU works,
# but note that CP shards on _sst alone, not on the largedim threshold.
#
# TT is likewise the MU solver on a different model family
# (tt_hybrid/README.md): Tucker factors, tensor-train core. It drops
# the same two flags as CP, for the same reasons (one kernel family; the factor
# update already l1-normalizes and absorbs the scale into that mode's TT core),
# and additionally:
#   --divergence fr      : the TT kernels are KL only; DIVERGENCES is forced to
#                          "kl" below and an fr request is reported, not run.
# BENCH_TT_RANK is the bond dimension. It is part of the artifact stem
# ('TT{tt_rank}b{order}D'), so sessions at different bonds never collide on disk
# and can be run back to back.
#
# UTT (uniform_tt/README.md) is TT with one core shared by every
# position. Same flags as TT (KL only, no --largedim / --normalize-factors), and
# it needs ALL modes tied, so it only runs entries whose tensor is all-tied.
# Note: at random init UTT can stall in a frequency-only state (flat error);
# BENCH_INIT=svd_stream avoids that, but then set it for every variant.
#
# SGD-TT is the SGD solver on the TT model (sgd/README.md, "TT and utt"). It
# takes everything from sgd (full data, ss1 keys, 20 iterations, random init)
# plus --decomposition tt and BENCH_TT_RANK, so it diffs row for row against an
# sgd session. One difference: its parametrization defaults to exp, the one
# sgd/README.md recommends for TT; sgd keeps softplus.
#
# All families report the SAME relative metric (rel_KL = KL/Σx,
# rel_FR = ||X-X̂||/||X||), so sessions on the same rows diff directly. What
# differs is the model, not the yardstick -- at equal rank CP is the strictly
# weaker family (a Tucker core has R^N free parameters where CP's λ has R) and
# TT sits between them (Σ_k ρ_k R_k ρ_{k+1} parameters, exact at large enough
# bond), so expect worse rec at equal rank and read the Δ as the price of the
# smaller model, not as a regression.
#
# Step mapping: one SGD loop "iteration" = SGD_STEPS_PER_ITER optimizer steps,
# so BENCH_ITERATIONS is NOT wall-clock comparable across solvers; compare total
# runtime and the rec/sem levels reached, not per-iteration cost. MU, CP and TT
# all run one full sweep per iteration, so their per-iteration times ARE
# comparable -- with the caveat that a TT core sweep is N sequential NNZ passes
# (one per site) where Tucker's is one.
#
# --- Per-run watchdog ---------------------------------------------------------
# MAX_SECONDS_PER_DECOMP (default 3600) caps ONE combo's decomposition process.
# On expiry the process is killed and the driver moves to the next combo,
# recording a TIMEOUT row (a timeout is not a result -- never diff it).
# One run per combo; CRASH recorded on any failure (e.g. OOM), TIMEOUT on expiry.

set -o pipefail

SOLVER="${BENCH_SOLVER:-tt}"
DRY_RUN="${BENCH_DRY_RUN:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        tt|utt|mu|sgd|sgd-tt|cp) SOLVER="$1"; shift ;;
        --solver)   SOLVER="$2"; shift 2 ;;
        --n-gpus)   BENCH_N_GPUS="$2"; shift 2 ;;
        --entries)  BENCH_ENTRIES="$2"; shift 2 ;;
        --name)     BENCH_NAME="$2"; shift 2 ;;
        --max-seconds) BENCH_MAX_SECONDS_PER_DECOMP="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        -h|--help)
            sed -n '3,33p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "ERROR: unknown argument '$1' (see --help)" >&2
            exit 2 ;;
    esac
done

case "$SOLVER" in
    tt|utt|mu|sgd|sgd-tt|cp) ;;
    *)
        echo "ERROR: variant must be tt, utt, mu, sgd, sgd-tt or cp, got '$SOLVER'" >&2
        exit 2 ;;
esac

# TT and UTT share the MU TT chain machinery (KL only).
IS_TT_FAMILY=false
[[ "$SOLVER" == "tt" || "$SOLVER" == "utt" ]] && IS_TT_FAMILY=true
# sgd and sgd-tt share the SGD settings; sgd-tt only adds the TT model.
IS_SGD=false
[[ "$SOLVER" == "sgd" || "$SOLVER" == "sgd-tt" ]] && IS_SGD=true

# FR is off by default (no longer used); BENCH_DIVERGENCES="kl fr" turns it back on.
read -r -a DIVERGENCES <<< "${BENCH_DIVERGENCES:-kl}"

# TT and UTT implement the KL/Poisson divergence only (the FR denominator needs
# a doubled, Gram-weighted chain). Drop anything else rather than queueing runs
# that are guaranteed to raise NotImplementedError at fit time.
if [[ "$IS_TT_FAMILY" == "true" ]]; then
    _tt_dropped=()
    for _d in "${DIVERGENCES[@]}"; do
        [[ "$_d" != "kl" ]] && _tt_dropped+=("$_d")
    done
    if (( ${#_tt_dropped[@]} > 0 )); then
        echo "NOTE: $SOLVER is KL only; skipping divergence(s): ${_tt_dropped[*]}" >&2
    fi
    DIVERGENCES=(kl)
fi
N_GPUS="${BENCH_N_GPUS:-2}"
# Init for every variant but sgd (which always starts random).
INIT="${BENCH_INIT:-random}"
# Free-text session label. Purely descriptive: it never touches run names or
# combo keys, so a labelled session still diffs against an unlabelled one.
NAME="${BENCH_NAME:-}"
RUN_FULL_DATA="${BENCH_FULL_DATA:-true}"
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

# --- SGD knobs (one config, so the matrix stays 1:1 with MU)
SGD_LR="${BENCH_SGD_LR:-1e-2}"
SGD_BATCH_SIZE="${BENCH_SGD_BATCH_SIZE:-4096}"
SGD_STEPS_PER_ITER="${BENCH_SGD_STEPS_PER_ITER:-100}"
# softplus | exp. exp is the recommended one for TT (sgd/README.md).
if [[ "$SOLVER" == "sgd-tt" ]]; then
    SGD_PARAMETRIZATION="${BENCH_SGD_PARAMETRIZATION:-exp}"
else
    SGD_PARAMETRIZATION="${BENCH_SGD_PARAMETRIZATION:-softplus}"
fi
SGD_WARM_START="${BENCH_SGD_WARM_START:-}"   # optional path to an MU model .pt used as init
# Cross-device collective for the sharded (N_GPUS > 1) path: auto | nccl | host.
# `auto` probes NCCL and falls back to pinned-host staging if it fails or hangs;
# `host` forces the fallback (slower, but always works — use it when a node's
# NCCL is broken and you need numbers); `nccl` refuses to fall back silently,
# which is what you want when a benchmark's timings must be NCCL timings.
SGD_COMM_BACKEND="${BENCH_SGD_COMM_BACKEND:-auto}"
# nnz per error (--sgd-eval-sample): every error, the recorded final one
# included, is measured on a fixed sample of this size (seeded by
# --random-state, so the same entries in every session with the same tensor and
# GPU count). An exact error over all nnz costs ~10x the whole training at
# order 4, rank 100 on the 1B tensor. Exception: SGD rows at an entry's
# full_data_ranks stay exact (one pass, at the last iteration), since they are
# compared with MU's exact errors. 0 = exact at every log step.
SGD_EVAL_SAMPLE="${BENCH_SGD_EVAL_SAMPLE:-1000000}"
if ! [[ "$SGD_EVAL_SAMPLE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: BENCH_SGD_EVAL_SAMPLE must be a non-negative integer, got '$SGD_EVAL_SAMPLE'" >&2
    exit 2
fi

# --- CP knobs (CP-APR); defaults match the kernel defaults, i.e. plain sweeps.
CP_INNER_ITERS="${BENCH_CP_INNER_ITERS:-1}"
CP_SCOOCH_KAPPA="${BENCH_CP_SCOOCH_KAPPA:-0.0}"

# --- TT / UTT knob: bond dimension of the tensor-train core (kernel default 100).
TT_RANK="${BENCH_TT_RANK:-100}"
if ! [[ "$TT_RANK" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: BENCH_TT_RANK must be a positive integer, got '$TT_RANK'" >&2
    exit 2
fi
# --- UTT knob: core-step exponent (float or "auto"); empty = code default (1).
UTT_CORE_POWER="${BENCH_UTT_CORE_POWER:-}"

if [[ "$IS_SGD" == "true" ]]; then
    ITERATIONS="${BENCH_ITERATIONS:-20}"
    # Reconstruction-error cadence (--rec-log-every). Deliberately > 1, unlike
    # MU, because the two solvers spend their time in different places: the exact
    # error pass is O(nnz * prod(rank)) for both, but for MU that is ~1/5 of an
    # iteration's work, while for SGD it is essentially the WHOLE iteration --
    # the SGD_STEPS_PER_ITER optimizer steps only touch
    # SGD_STEPS_PER_ITER * SGD_BATCH_SIZE entries (409,600 at the defaults), so
    # at --rec-log-every 1 the benchmark would be timing full_relative_error, not
    # the solver. (SGD_EVAL_SAMPLE makes these in-loop errors cheap.) Keep
    # LOG_EVERY a divisor of ITERATIONS: rec_errors only receives samples at the
    # log cadence, so a non-divisor reports an intermediate iterate as the last.
    LOG_EVERY="${BENCH_LOG_EVERY:-5}"
    SEM_CHECK_EVERY="$ITERATIONS"
    [[ "$SOLVER" == "sgd-tt" ]] && RUN_PREFIX="bench_sgdtt_" || RUN_PREFIX="bench_sgd_"
else
    # MU, CP, TT, UTT: one full sweep per iteration, so the rows line up 1:1.
    ITERATIONS="${BENCH_ITERATIONS:-4}"
    LOG_EVERY="${BENCH_LOG_EVERY:-1}"
    SEM_CHECK_EVERY=4
    case "$SOLVER" in
        mu) RUN_PREFIX="bench_" ;;
        *)  RUN_PREFIX="bench_${SOLVER}_" ;;
    esac
fi

DATA_DIR="${SCRATCH_DATA:-$DATA}"
if [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: neither SCRATCH_DATA nor DATA is set" >&2
    exit 1
fi

# Entries are passed to nnt as --dataset "$FOLDER/<entry>", i.e. relative to
# $DATA/tensors, so the folder must live there.
FOLDER="${BENCH_FOLDER:-benchmarking}"
ROOT="$DATA_DIR/tensors/$FOLDER"

# Settings file: default is the one in the repo (this file is src/tensormet/scripts/).
SETTINGS="${BENCH_SETTINGS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/2_benchmarking/benchmark_settings.conf}"
if [[ ! -f "$SETTINGS" ]]; then
    echo "ERROR: no settings file at $SETTINGS" >&2
    echo "       set BENCH_SETTINGS to 2_benchmarking/benchmark_settings.conf" >&2
    exit 2
fi

BENCHMARK_JSON="${BENCH_JSON:-$DATA_DIR/benchmarking/benchmark.json}"
OLD_JSON="${BENCHMARK_JSON%.json}_old.json"
mkdir -p "$(dirname "$BENCHMARK_JSON")"

# --- Entries --------------------------------------------------------------------

# _read_entry <entry>: fills E_* from <entry>/entry.conf (what the tensor is).
# Parsed line by line (never sourced); other keys are ignored. Returns 1 with a
# message if the file or a key is missing.
_read_entry() {
    local conf="$ROOT/$1/entry.conf" line key val
    E_ORDER=""; E_SHARED_FACTORS=""; E_DIM=""; E_METHOD=""
    if [[ ! -f "$conf" ]]; then
        echo "ERROR: no entry.conf in $ROOT/$1" >&2
        return 1
    fi
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        [[ -z "${line// }" || "$line" == \#* ]] && continue
        key="${line%%=*}"; val="${line#*=}"
        val="${val%\"}"; val="${val#\"}"
        case "$key" in
            ORDER)           E_ORDER="$val" ;;
            SHARED_FACTORS)  E_SHARED_FACTORS="$val" ;;
            DIM)             E_DIM="$val" ;;
            METHOD)          E_METHOD="$val" ;;
        esac
    done < "$conf"
    local k
    for k in ORDER SHARED_FACTORS DIM METHOD; do
        local v="E_$k"
        if [[ -z "${!v}" ]]; then
            echo "ERROR: $conf has no $k" >&2
            return 1
        fi
    done
    # The entry's tensor and vocab (one of each per entry).
    E_TENSOR="$(ls "$ROOT/$1/populated/${E_METHOD}_${E_ORDER}D_${E_DIM}d"*.pt 2>/dev/null | head -1)"
    E_VOCAB="$(ls "$ROOT/$1/vocabularies/${E_ORDER}D_${E_DIM}d"*.pkl 2>/dev/null | head -1)"
}

# Settings lines: "<entry> <subsample> <ranks> <full_data_ranks> <variants>".
# _settings_entries: the entry names in the settings file, in file order.
_settings_entries() {
    local name rest
    while read -r name rest || [[ -n "$name" ]]; do
        name="${name%$'\r'}"
        [[ -z "$name" || "$name" == \#* ]] && continue
        echo "$name"
    done < "$SETTINGS"
}

# _read_settings <entry>: fills S_SUBSAMPLE, S_RANKS, S_FULL_DATA_RANKS and
# S_VARIANTS (space-separated). Returns 1 if the entry has no line, 2 if its
# line is malformed.
_read_settings() {
    local name ss ranks full variants extra v
    while read -r name ss ranks full variants extra || [[ -n "$name" ]]; do
        [[ "$name" != "$1" ]] && continue
        variants="${variants%$'\r'}"
        if [[ -z "$variants" || -n "$extra" ]]; then
            echo "ERROR: $SETTINGS: the line for $1 needs 5 columns (entry subsample ranks full_data_ranks variants)" >&2
            return 2
        fi
        S_SUBSAMPLE="$ss"
        S_RANKS="${ranks//,/ }"
        [[ "$full" == "-" ]] && S_FULL_DATA_RANKS="" || S_FULL_DATA_RANKS="${full//,/ }"
        S_VARIANTS="${variants//,/ }"
        for v in $S_VARIANTS; do
            case "$v" in
                tt|utt|mu|sgd|sgd-tt|cp) ;;
                *) echo "ERROR: $SETTINGS: unknown variant '$v' for $1" >&2; return 2 ;;
            esac
        done
        return 0
    done < "$SETTINGS"
    return 1
}

# _norm_frac <frac>: "1", "1.0", "1.00" -> "1", so full-data keys always match.
_norm_frac() {
    [[ "$1" =~ ^1(\.0*)?$ ]] && echo 1 || echo "$1"
}

# _skip_reason <entry>: why this variant does not run the entry ("" = it runs).
_skip_reason() {
    if [[ " $S_VARIANTS " != *" $SOLVER "* ]]; then
        echo "$SOLVER is not among its variants in the settings file ($S_VARIANTS)"
        return
    fi
    if [[ "$SOLVER" == "utt" ]]; then
        # All modes tied <=> the file suffix is _shared01..(order-1).
        local all_tied="_shared$(seq -s '' 0 $((E_ORDER - 1)))"
        if [[ "$E_TENSOR" != *"${all_tied}.pt" ]]; then
            echo "utt needs all modes tied (SHARED_FACTORS=$E_SHARED_FACTORS)"
        fi
    fi
}

mapfile -t _listed < <(_settings_entries)
if [[ -n "${BENCH_ENTRIES:-}" ]]; then
    read -r -a ENTRIES <<< "$BENCH_ENTRIES"
else
    ENTRIES=("${_listed[@]}")
fi
if (( ${#ENTRIES[@]} == 0 )); then
    echo "ERROR: no entries listed in $SETTINGS" >&2
    exit 1
fi
# Entries in the folder that the settings file does not mention.
for _conf in "$ROOT"/*/entry.conf; do
    [[ -f "$_conf" ]] || continue
    _e="$(basename "$(dirname "$_conf")")"
    [[ " ${_listed[*]} " != *" $_e "* ]] && echo "NOTE: entry $_e has no line in $SETTINGS; not run" >&2
done

# Pre-flight: every entry has settings, is readable and complete; decide which
# ones this variant runs.
RUN_ENTRIES=()
_preflight_failed=false
for entry in "${ENTRIES[@]}"; do
    _read_settings "$entry"
    case $? in
        1) echo "ERROR: entry $entry has no line in $SETTINGS" >&2
           _preflight_failed=true; continue ;;
        2) _preflight_failed=true; continue ;;
    esac
    if [[ ! -d "$ROOT/$entry" ]]; then
        echo "ERROR: entry $entry is in $SETTINGS but not in $ROOT" >&2
        echo "       (create it with python -m tensormet.scripts.add_benchmark_entry)" >&2
        _preflight_failed=true; continue
    fi
    if ! _read_entry "$entry"; then
        _preflight_failed=true; continue
    fi
    _missing=()
    [[ -z "$E_TENSOR" ]] && _missing+=("populated/${E_METHOD}_${E_ORDER}D_${E_DIM}d*.pt")
    [[ -z "$E_VOCAB" ]] && _missing+=("vocabularies/${E_ORDER}D_${E_DIM}d*.pkl")
    compgen -G "$ROOT/$entry/eval_sentences/*.parquet" >/dev/null || _missing+=("eval_sentences/*.parquet")
    if (( ${#_missing[@]} > 0 )); then
        echo "ERROR: entry $entry is missing: ${_missing[*]}" >&2
        _preflight_failed=true; continue
    fi
    _why="$(_skip_reason "$entry")"
    if [[ -n "$_why" ]]; then
        echo "NOTE: skipping entry $entry: $_why" >&2
        continue
    fi
    RUN_ENTRIES+=("$entry")
done
[[ "$_preflight_failed" == "true" ]] && exit 1
if (( ${#RUN_ENTRIES[@]} == 0 )); then
    echo "ERROR: no entry in $ROOT runs with variant $SOLVER" >&2
    exit 1
fi

# Old log: sessions without "protocol" are moved to benchmark_old.json at save
# time. Stop now (not after hours of runs) if that would overwrite an old file.
if [[ "$DRY_RUN" != "true" && -f "$BENCHMARK_JSON" ]]; then
    python3 - "$BENCHMARK_JSON" "$OLD_JSON" <<'PYEOF' || exit 2
import json, sys
from pathlib import Path
new, old = sys.argv[1:]
data = json.loads(Path(new).read_text())
data = data if isinstance(data, list) else [data]
if any("protocol" not in s for s in data) and Path(old).exists():
    sys.exit(f"ERROR: {new} holds old-style sessions and {old} already exists.\n"
             f"       Merge or rename one of them by hand first.")
PYEOF
fi

if (( N_GPUS > 1 )); then
    if [[ "$IS_SGD" == "true" ]]; then
        echo "WARNING: N_GPUS=$N_GPUS — the sharded SGD path (ShardedSGDTrainer) will be taken;" >&2
        echo "         timings are not comparable to single-GPU runs, and the effective batch" >&2
        echo "         is SGD_BATCH_SIZE per shard." >&2
    else
        echo "WARNING: N_GPUS=$N_GPUS — sharded multi-GPU path will be taken; timings are not comparable to single-GPU runs." >&2
    fi
fi

[[ -n "$NAME" ]] && echo "benchmark name:   $NAME"
echo "benchmark config: variant=$SOLVER n_gpus=$N_GPUS iterations=$ITERATIONS init=$([[ "$IS_SGD" == "true" ]] && echo random || echo "$INIT")"
if (( MAX_SECONDS_PER_DECOMP > 0 )); then
    echo "                  per-run limit=${MAX_SECONDS_PER_DECOMP}s (killed and recorded TIMEOUT on expiry)"
else
    echo "                  per-run limit=off"
fi
echo "                  entries=[${RUN_ENTRIES[*]}] from $ROOT"
echo "                  settings=$SETTINGS"
echo "                  ranks=[${BENCH_RANKS:-per entry}] divergences=[${DIVERGENCES[*]}]"
if [[ "$IS_SGD" == "true" ]]; then
    echo "                  sgd=adam/$SGD_PARAMETRIZATION lr=$SGD_LR bs=$SGD_BATCH_SIZE steps/iter=$SGD_STEPS_PER_ITER eval_sample=$SGD_EVAL_SAMPLE"
    [[ "$SOLVER" == "sgd-tt" ]] && echo "                  model=Tucker-TT hybrid, tt_rank=$TT_RANK"
    if (( N_GPUS > 1 )); then
        echo "                  comm_backend=$SGD_COMM_BACKEND (resolved backend is logged per run)"
    fi
elif [[ "$SOLVER" == "cp" ]]; then
    echo "                  cp=EXPERIMENTAL nonnegative CP, inner_iters=$CP_INNER_ITERS scooch_kappa=$CP_SCOOCH_KAPPA"
elif [[ "$SOLVER" == "tt" ]]; then
    echo "                  tt=Tucker-TT hybrid, tt_rank=$TT_RANK (KL only)"
elif [[ "$SOLVER" == "utt" ]]; then
    echo "                  utt=shared-core TT, tt_rank=$TT_RANK core_power=${UTT_CORE_POWER:-1} (KL only, all modes tied)"
fi
echo "                  results -> $BENCHMARK_JSON"

TMPJSON=$(mktemp "${TMPDIR:-/tmp}/bench_${SOLVER}_results_XXXXXX.json")
echo '{}' > "$TMPJSON"
trap "rm -f $TMPJSON" EXIT

_cleanup_run() {
    local decomp_dir="$1"
    local run_name="$2"
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

# SGD, CP and TT artifacts carry their own stem fragment ('SGD{order}D' /
# 'CP{order}D' / 'TT{tt_rank}b{order}D'), but run_name is still the leading
# prefix, so one set of globs covers all solvers.
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
final_error = None  # the run's last logged error (SGD: on the eval sample unless exact)
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
        final_error = timing.get('final_error')
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
    "final_error": final_error,
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

# _build_args <entry> <rank> <div> <subsample>
# Fills the global NNT_ARGS array from the E_* settings of the current entry:
# shared flags first, then the solver-specific tail.
_build_args() {
    local entry="$1" rank="$2" div="$3" subsample="$4"
    local shared="$E_SHARED_FACTORS"
    # UTT shares one core across positions: every mode tied (checked in pre-flight).
    [[ "$SOLVER" == "utt" ]] && shared="all"
    # SGD rows that must be exact (SGD_EXACT_ROW, set by the matrix loop): no eval
    # sample, and the error logged once, at the last iteration.
    local log_every="$LOG_EVERY"
    [[ "$IS_SGD" == "true" && "$SGD_EXACT_ROW" == "true" ]] && log_every="$ITERATIONS"

    NNT_ARGS=(
        --dataset "$FOLDER/$entry"
        --method "$E_METHOD"
        --divergence "$div"
        --name "$RUN_NAME"
        --dim "$E_DIM"
        --order "$E_ORDER"
        --rank "$rank"
        --verbose t
        --iterations "$ITERATIONS"
        --checkpoint-saving-steps 0
        --max-cpu-frac 0.8
        --sem-error-type all
        --rec-log-every "$log_every"
        --sem-check-every "$SEM_CHECK_EVERY"
        --shared-factors "$shared"
        --sem-primary-key simlex_all_rho
        --n-gpus "$N_GPUS"
        --random-state 1
    )

    if [[ "$IS_SGD" == "true" ]]; then
        # --subsample-frac 1.0 is explicit: the config default is 0.025 (since
        # 2026-09-15), and SGD refuses anything below 1.
        NNT_ARGS+=(
            --solver sgd
            --objective full
            --subsample-frac 1.0
            --init random
            --sgd-lr "$SGD_LR"
            --sgd-batch-size "$SGD_BATCH_SIZE"
            --sgd-parametrization "$SGD_PARAMETRIZATION"
            --sgd-steps-per-iteration "$SGD_STEPS_PER_ITER"
            --sgd-comm-backend "$SGD_COMM_BACKEND"
        )
        if (( SGD_EVAL_SAMPLE > 0 )) && [[ "$SGD_EXACT_ROW" != "true" ]]; then
            NNT_ARGS+=(--sgd-eval-sample "$SGD_EVAL_SAMPLE")
        fi
        [[ "$SOLVER" == "sgd-tt" ]] && NNT_ARGS+=(--decomposition tt --tt-rank "$TT_RANK")
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
    elif [[ "$IS_TT_FAMILY" == "true" ]]; then
        # No --largedim / --normalize-factors, same reasons as CP.
        NNT_ARGS+=(
            --decomposition "$SOLVER"
            --tt-rank "$TT_RANK"
            --objective full
            --subsample-frac "$subsample"
        )
        if [[ "$SOLVER" == "utt" && -n "$UTT_CORE_POWER" ]]; then
            NNT_ARGS+=(--utt-core-power "$UTT_CORE_POWER")
        fi
    else
        NNT_ARGS+=(
            --largedim true
            --normalize-factors t
            --subsample-frac "$subsample"
        )
    fi

    # Same init for every variant; sgd and sgd-tt set their own (random) above.
    [[ "$IS_SGD" != "true" ]] && NNT_ARGS+=(--init "$INIT")
}

# _run_one <entry> <rank> <div> <subsample>   (E_* must hold the entry's settings)
_run_one() {
    local entry="$1" rank="$2" div="$3" subsample
    subsample="$(_norm_frac "$4")"
    local key="${entry}__rank${rank}__${div}__ss${subsample}"
    local decomp_dir="$ROOT/$entry/decomposition"
    RUN_NAME="${RUN_PREFIX}rank${rank}__${div}__ss${subsample}"

    _build_args "$entry" "$rank" "$div" "$subsample"

    if [[ "$IS_SGD" == "true" ]]; then
        local err="sample $SGD_EVAL_SAMPLE"
        [[ "$SGD_EXACT_ROW" == "true" || "$SGD_EVAL_SAMPLE" == 0 ]] && err="exact"
        echo -e "\n\n\n>>> [$key] (solver=$SOLVER param=$SGD_PARAMETRIZATION lr=$SGD_LR bs=$SGD_BATCH_SIZE error=$err)"
    elif [[ "$SOLVER" == "cp" ]]; then
        echo -e "\n\n\n>>> [$key] (solver=cp subsample=$subsample inner_iters=$CP_INNER_ITERS)"
    elif [[ "$IS_TT_FAMILY" == "true" ]]; then
        echo -e "\n\n\n>>> [$key] (solver=$SOLVER subsample=$subsample tt_rank=$TT_RANK)"
    else
        echo -e "\n\n\n>>> [$key] (solver=mu subsample=$subsample)"
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

    _cleanup_run "$decomp_dir" "$RUN_NAME"
}

# --- The matrix -----------------------------------------------------------------
# Per entry: its ranks at its subsample (SGD: full data), then the
# full_data_ranks rows at subsample 1. SGD runs full data anyway, so for SGD
# these are its rows at those ranks with an exact error (added if not in ranks).
# ENTRY_META records each entry's settings and tensor fingerprint in the session.
ENTRY_META=""
for entry in "${RUN_ENTRIES[@]}"; do
    _read_entry "$entry" || exit 1
    _read_settings "$entry" || exit 1
    ENTRY_META+="${entry}|order=${E_ORDER} shared=${E_SHARED_FACTORS} dim=${E_DIM} method=${E_METHOD}"
    ENTRY_META+=" ss=${S_SUBSAMPLE} tensor=$(stat -c %s "$E_TENSOR")@$(stat -c %Y "$E_TENSOR")"$'\n'

    read -r -a _ranks <<< "${BENCH_RANKS:-$S_RANKS}"
    read -r -a _full_ranks <<< "$S_FULL_DATA_RANKS"
    if [[ "$IS_SGD" == "true" ]]; then
        _ss=1
        if [[ "$RUN_FULL_DATA" == "true" ]]; then
            for rank in "${_full_ranks[@]}"; do
                [[ " ${_ranks[*]} " != *" $rank "* ]] && _ranks+=("$rank")
            done
        fi
    else
        _ss="$(_norm_frac "$S_SUBSAMPLE")"
    fi

    for rank in "${_ranks[@]}"; do
        # SGD at a full_data rank: exact error, so it lines up with MU's ss1 row.
        SGD_EXACT_ROW=false
        if [[ "$IS_SGD" == "true" && "$RUN_FULL_DATA" == "true" && " ${_full_ranks[*]} " == *" $rank "* ]]; then
            SGD_EXACT_ROW=true
        fi
        for div in "${DIVERGENCES[@]}"; do
            _run_one "$entry" "$rank" "$div" "$_ss"
        done
    done
    SGD_EXACT_ROW=false

    if [[ "$IS_SGD" != "true" && "$RUN_FULL_DATA" == "true" && "$_ss" != "1" ]]; then
        for rank in "${_full_ranks[@]}"; do
            for div in "${DIVERGENCES[@]}"; do
                _run_one "$entry" "$rank" "$div" 1
            done
        done
    fi
done

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
        if val.get('final_error') is not None:
            final_rec = val['final_error']
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

# Append results to benchmark.json (one file for every solver, tagged by session)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
SGD_META="$(printf '%s|%s|%s|%s|%s' "$SGD_PARAMETRIZATION" "$SGD_LR" "$SGD_BATCH_SIZE" "$SGD_STEPS_PER_ITER" "$SGD_EVAL_SAMPLE")"
CP_META="$(printf '%s|%s' "$CP_INNER_ITERS" "$CP_SCOOCH_KAPPA")"
python3 - "$TIMESTAMP" "$N_GPUS" "$TMPJSON" "$BENCHMARK_JSON" "$SOLVER" "$SGD_META" "$NAME" "$CP_META" "$TT_RANK" "${UTT_CORE_POWER:-1}" "$INIT" "$FOLDER" "$ENTRY_META" "$OLD_JSON" <<'PYEOF'
import json, sys
from pathlib import Path

(timestamp, n_gpus, tmpjson, benchmark_json, solver, sgd_meta, name,
 cp_meta, tt_rank, utt_core_power, init, folder, entry_meta, old_json) = sys.argv[1:]

with open(tmpjson) as f:
    results = json.load(f)

# Metadata keys must stay scalars/strings, except "entries";
# benchmark_inspections.ipynb lists every metadata key in _BENCH_META_KEYS.
entry = {
    "time": timestamp,
    "protocol": 2,
    "folder": folder,
    "n_gpus": int(n_gpus),
    "solver": solver,
    "entries": dict(line.split("|", 1) for line in entry_meta.splitlines() if line),
}
# Session-level method for the notebook's Method filter (one per entry).
entry["method"] = ",".join(sorted({part.split("=", 1)[1]
                                   for meta in entry["entries"].values()
                                   for part in meta.split() if part.startswith("method=")}))
if name:
    entry["name"] = name
if solver not in ("sgd", "sgd-tt") and init != "random":
    entry["init"] = init
if solver in ("sgd", "sgd-tt"):
    parametrization, lr, batch_size, steps, eval_sample = sgd_meta.split("|")
    entry["sgd_config"] = f"adam/{parametrization} lr={lr} bs={batch_size} steps/iter={steps}"
    if eval_sample != "0":
        entry["sgd_config"] += f" eval_sample={eval_sample}"
if solver == "sgd-tt":
    entry["tt_config"] = f"tt_rank={tt_rank}"
elif solver == "cp":
    inner_iters, scooch_kappa = cp_meta.split("|")
    entry["cp_config"] = f"inner_iters={inner_iters} scooch_kappa={scooch_kappa}"
elif solver == "tt":
    entry["tt_config"] = f"tt_rank={tt_rank}"
elif solver == "utt":
    entry["utt_config"] = f"tt_rank={tt_rank} core_power={utt_core_power} shared=all"
entry.update(results)

path, old = Path(benchmark_json), Path(old_json)
data = []
if path.exists():
    data = json.loads(path.read_text())
    data = data if isinstance(data, list) else [data]
    # Sessions from before the benchmark folder: move the whole file aside once.
    if any("protocol" not in s for s in data):
        if old.exists():
            keep = path.with_name(f"benchmark_unsaved_{timestamp.replace(':', '')}.json")
            keep.write_text(json.dumps([entry], indent=2))
            sys.exit(f"ERROR: {path} holds old-style sessions and {old} exists; "
                     f"this session was saved to {keep} instead.")
        path.rename(old)
        print(f"Moved the old sessions to {old}")
        data = []

data.append(entry)
path.write_text(json.dumps(data, indent=2))
print(f"Saved results to {benchmark_json}")
PYEOF
