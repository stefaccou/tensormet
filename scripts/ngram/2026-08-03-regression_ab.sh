#!/bin/bash
# Regression-bisect runner: trimmed copy of 2026-06-22-fineweb_raw_100M_subsampled.sh.
# Runs ONE fast decomposition (countingLog, 20 iterations, no semantic eval, no
# checkpoint I/O) so per-iteration speed can be compared across code versions.
#
# usage:
#   bash scripts/ngram/2026-08-03-regression_ab.sh <tag>                        # current code
#   bash scripts/ngram/2026-08-03-regression_ab.sh <tag> regression_code/<tag>  # old code
#
#   <tag>          short label, e.g. now / jul06 / jul02. Artifacts get the name
#                  regr_<tag>, keeping every bisect rung separate.
#   second arg     optional: a snapshot folder from regression_code/ holding an
#                  old version of the code (pre-extracted from git history; see
#                  reviews/2026-08-03_bisect-walkthrough.md). Its src/ is put on
#                  PYTHONPATH so python imports THAT version of tensormet instead
#                  of the installed one. The script verifies the import really
#                  comes from the snapshot and ABORTS if not — a run can no
#                  longer silently test the wrong code.
#
# Compare across tags: the printed "time=" lines (median, skip the first) and the
# final "TOTAL WALL TIME" line. Wall time is the safest cross-version metric —
# the per-iteration time= changed meaning in the 2026-08-03 code (it now
# measures true execution instead of kernel queueing).

set -euo pipefail

TAG="${1:?usage: bash $0 <tag> [snapshot-dir]}"

if [[ $# -ge 2 ]]; then
    WT="$(realpath "$2")"
    if [[ ! -d "$WT/src/tensormet" ]]; then
        echo "ERROR: $WT/src/tensormet does not exist - is '$2' really a snapshot folder?" >&2
        exit 1
    fi
    export PYTHONPATH="$WT/src"
fi

CODE_FILE="$(python3 -c 'import tensormet, inspect; print(inspect.getfile(tensormet))')"
echo "=== code under test: $CODE_FILE"
if [[ $# -ge 2 && "$CODE_FILE" != "$WT"* ]]; then
    echo "ERROR: python imported tensormet from OUTSIDE the snapshot shown above." >&2
    echo "       Aborting so the run does not measure the wrong code." >&2
    exit 1
fi

DATASET_100M="4-gram-raw-bos-eos-fineweb-en_100000000"

t0=$SECONDS
python3 -m tensormet.scripts.nnt \
    --dataset "$DATASET_100M" \
    --method countingLog \
    --name "regr_${TAG}" \
    --divergence kl \
    --dim 10000 \
    --order 4 \
    --rank 100 \
    --shared-factors "0-1,1-2,2-3" \
    --verbose f \
    --max-cpu-frac 0.8 \
    --patience 20 \
    --iterations 20 \
    --rec-log-every 2 \
    --largedim true \
    --sem-check-every 1000 \
    --sem-fitness-target 10000 \
    --sem-error-type all \
    --sem-primary-key simlex_all_rho \
    --checkpoint-saving-steps 1000 \
    --random-state 1 \
    --normalize-factors true \
    --return-errors full \
    --overwrite t \
    --resume f \
    --n-gpus 4 \
    --subsample-frac 0.01

echo "=== TOTAL WALL TIME for tag '${TAG}': $((SECONDS - t0)) s (20 iterations + data load)"
