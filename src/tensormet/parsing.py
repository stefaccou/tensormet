"""CLI parsing helpers for constructing a RunConfig from defaults + overrides.

This module provides a single function `parse_run_config(argv=None)` which:
- Instantiates the default RunConfig using the defaults from tensormet.config
- Parses command-line arguments (if present) and overrides only the provided
  values on the respective dataclasses (ExperimentConfig, TrainingConfig,
  EvalConfig)

The module is intentionally self-contained so external launcher scripts can do:

from tensormet.parsing import parse_run_config
cfg = parse_run_config()
# then pass cfg into whatever runner function you have

When run as a script it will print the resulting config JSON to stdout.
"""
from __future__ import annotations
from dataclasses import replace, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import json

from tensormet.config import (
    ExperimentConfig,
    TrainingConfig,
    EvalConfig,
    RunConfig,
    VectorExperimentConfig,
    VectorRunConfig,
    HFStreamConfig,
    PopulationExperimentConfig,
    PopulationRunConfig,
    parse_ngram_order,
    parse_raw_ngram_order,
)


def _parse_bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    s2 = str(s).lower()
    if s2 in ("1", "true", "t", "yes", "y"):
        return True
    if s2 in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {s}")


def _parse_rank(s: str, n_modes: Optional[int] = None) -> Tuple[int, ...]:
    # Accept comma-separated integers like "100,100,100,100" or a single int
    if not s:
        return tuple()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid rank specification: {s}")

    if n_modes is not None and len(vals) == 1:
        vals = vals * n_modes
    return vals


def _parse_shared_factors(s: str):
    """
    Parse shared factor links.

    Accepts:
      --shared-factors none
      --shared-factors all
      --shared-factors 1-2
      --shared-factors 1-2,2-0
      --shared-factors 1:2,2:0

    Returns:
      None   (if 'none'/'null'/'')
      "all"  (sentinel — resolved to all pairs after order is known)
      tuple(((a,b), ...))
    """
    if s is None:
        return None
    s2 = str(s).strip().lower()
    if s2 in ("", "none", "null", "no"):
        return None
    if s2 == "all":
        return "all"

    pairs = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
        elif ":" in token:
            a, b = token.split(":", 1)
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid --shared-factors token '{token}'. Use like '1-2,2-0' or 'none'."
            )
        try:
            ai = int(a.strip())
            bi = int(b.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid --shared-factors token '{token}': indices must be ints."
            )
        if ai == bi:
            raise argparse.ArgumentTypeError(
                f"Invalid --shared-factors token '{token}': cannot link a mode to itself."
            )
        pairs.add(tuple(sorted((ai, bi))))

    return tuple(sorted(pairs)) if pairs else None

def _parse_cols_to_build(s: str) -> Tuple[str, ...]:
    if not s:
        return tuple()
    cols = tuple(part.strip() for part in s.split(",") if part.strip())
    if not cols:
        raise argparse.ArgumentTypeError("Invalid cols_to_build specification.")
    return cols

def _none_if_missing(value, sentinel=None):
    # Helper: treat argparse's default sentinel as missing -> return None
    return None if value is sentinel else value


def _parse_gpu_id(s: str):
    """
    Parse --gpu-id as a single int or a comma-separated list of ints.

      "0"     → 0          (single GPU, stays an int for backwards compat)
      "0,1"   → (0, 1)     (multi-GPU explicit pin)
      "0,1,2" → (0, 1, 2)
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid --gpu-id value: {s!r}. Expected int or comma-separated ints.")
    if not vals:
        raise argparse.ArgumentTypeError(f"Invalid --gpu-id value: {s!r}")
    return vals[0] if len(vals) == 1 else vals


def _parse_ensured_vocab(s: str) -> Tuple[str, ...]:
    """Parse a comma-separated list of token strings, e.g. '<BOS>,<EOS>'."""
    if not s:
        return tuple()
    return tuple(tok for tok in s.split(",") if tok)


def _parse_top_ks(s: str) -> Tuple[int, ...]:
    if not s:
        return tuple()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid top-ks specification: {s}")


def _parse_top_ks_asymmetric(s: str) -> Tuple[Tuple[int, ...], ...]:
    """Parse per-mode vocab-size variants separated by '|', modes by ','.

    Example: "1000,2000,1000|2000,4000,2000"
      → ((1000, 2000, 1000), (2000, 4000, 2000))
    """
    if not s:
        return tuple()
    variants = []
    for variant_str in s.split("|"):
        parts = [p.strip() for p in variant_str.split(",") if p.strip()]
        if not parts:
            continue
        try:
            variants.append(tuple(int(p) for p in parts))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid --top-ks-asymmetric variant {variant_str!r}. "
                "Expected comma-separated ints, variants separated by '|'."
            )
    return tuple(variants)


def _parse_dim(s: str) -> "int | Tuple[int, ...]":
    """Parse --dim as a single int or comma-separated ints (for asymmetric tensors).

    "1000"           → 1000
    "1000,2000,1000" → (1000, 2000, 1000)  — uniform collapses back to int
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid --dim value: {s!r}. Expected int or comma-separated ints.")
    if len(vals) == 1:
        return vals[0]
    if len(set(vals)) == 1:
        return vals[0]
    return vals


def parse_run_config(argv: Optional[List[str]] = None) -> RunConfig:
    """Parse CLI args and return a RunConfig built from defaults with overrides.

    Args:
        argv: optional list of arguments (like sys.argv[1:]). If None, argparse
              will read from the actual command line.
    Returns:
        RunConfig with overrides applied only for flags provided by the user.
    """
    # Build defaults from the dataclasses defined in config.py
    default_exp = ExperimentConfig()
    default_train = TrainingConfig()
    default_eval = EvalConfig()

    parser = argparse.ArgumentParser(description="Build a RunConfig from defaults and CLI overrides")

    # Experiment-level args
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--divergence", type=str, default=None)
    parser.add_argument("--dim", type=_parse_dim, default=None)
    parser.add_argument("--order", type=int, default=None) # new: order
    parser.add_argument("--rank", type=str, default=None,
                        help="Comma-separated ranks, e.g. --rank 100,100,100 or single int")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--random-state", type=int, dest="random_state", default=None)
    parser.add_argument("--max-cpu-frac", type=float, default=None)
    parser.add_argument("--tier1", type=_parse_bool, default=None)
    parser.add_argument("--overwrite", type=_parse_bool, default=None)
    parser.add_argument("--data-dir", type=Path, dest="data_dir", default=None)

    # Training-level args
    parser.add_argument("--iterations", type=int, dest="n_iter_max", default=None,
                        help="Alias for --n-iter-max")
    parser.add_argument("--n-iter-max", type=int, dest="n_iter_max", default=None,
                        help="Maximum number of training iterations")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--init", type=str, default=None)
    parser.add_argument("--normalize-factors", type=_parse_bool, default=None,
                        help="true/false")
    # new: factor sharing
    parser.add_argument(
        "--shared-factors",
        type=_parse_shared_factors,
        default=None,
        help="Factor linking, e.g. --shared-factors 1-2,2-0 or 'all' to share all. Use 'none' to disable.",
    )
    parser.add_argument("--warmup-steps", type=int, dest="warmup_steps", default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--verbose", type=_parse_bool, default=None)
    parser.add_argument("--return-errors", type=str, dest="return_errors", default=None)
    parser.add_argument("--largedim", type=_parse_bool, default=None)
    parser.add_argument("--checkpoint-saving-steps", type=int, dest="checkpoint_saving_steps", default=None)
    # NEW: Checkpoint resumption flag
    parser.add_argument("--resume", type=_parse_bool, default=None,
                        help="Resume training from the latest available checkpoint for this configuration.")
    # NEW: multi-gpu
    parser.add_argument("--hpc", type=_parse_bool, default=None,
                        help="Stage per-run artifact writes to node-local $TMPDIR during the "
                             "run, copying back to data_dir at the end. Relieves shared-GPFS "
                             "write contention when many array tasks run concurrently.")
    parser.add_argument("--n-gpus", type=int, dest="n_gpus", default=None)
    parser.add_argument("--gpu-id", type=_parse_gpu_id, dest="gpu_id", default=None,
                        help="Physical GPU(s) to pin, e.g. --gpu-id 0 or --gpu-id 0,1,2")
    parser.add_argument("--subsample-frac", type=float, dest="subsample_frac", default=None)
    parser.add_argument("--max-nnz", type=int, dest="max_nnz", default=None,
                        help="Hard ceiling on NNZ entries used per update step (global across "
                             "all GPU shards). Combines with --subsample-frac as "
                             "min(round(frac*nnz), max_nnz). 0 disables. Note: "
                             "--subsample-warmup iterations still use the full tensor.")
    parser.add_argument("--subsample-warmup", type=int, dest="subsample_warmup", default=None)
    parser.add_argument("--objective", type=str, default=None,
                        choices=["full", "masked"],
                        help="'full' fits the zero-filled tensor (count data); "
                             "'masked' fits only observed entries (completion/rating data).")
    # EXPERIMENTAL (reviews/CP_IMPLEMENTATION_PLAN.md): CP decomposition family.
    parser.add_argument("--decomposition", type=str, default=None,
                        choices=["tucker", "cp"],
                        help="Decomposition family: 'tucker' (default, existing pipeline) or "
                             "'cp' (EXPERIMENTAL nonnegative CP; single-GPU, objective=full only).")
    parser.add_argument("--cp-inner-iters", type=int, dest="cp_inner_iters", default=None,
                        help="CP only: CP-APR inner iterations per mode per sweep (default 1).")
    parser.add_argument("--cp-scooch-kappa", type=float, dest="cp_scooch_kappa", default=None,
                        help="CP only: CP-APR 'scooch' nudge for inadmissible zeros (default 0 = off).")

    # Eval-level args
    parser.add_argument("--rec-check-every", type=int, dest="rec_check_every", default=None)
    parser.add_argument("--rec-log-every", type=int, dest="rec_log_every", default=None)
    parser.add_argument("--sem-check-every", type=int, dest="sem_check_every", default=None)
    parser.add_argument("--sem-error-type", type=str, dest="sem_error_type", default=None)
    parser.add_argument("--sem-primary-key", type=str, dest="sem_primary_key", default=None,
                        help="Override the metric used for patience/diff, e.g. simlex_all_rho")
    parser.add_argument("--sem-softmax-temperature", type=float, dest="sem_softmax_temperature", default=None)
    parser.add_argument("--sem-fitness-target", type=int, dest="sem_fitness_target", default=None)
    parser.add_argument("--dim-consistency", type=_parse_bool, dest="dim_consistency", default=None,
                        help="Enable LLM-as-judge dimension-consistency scoring at each semantic "
                             "check (default: false). Loads a small judge model (~1 GB fp16) on "
                             "the GPU at the first check.")
    parser.add_argument("--dim-consistency-words", type=int, dest="dim_consistency_words", default=None,
                        help="Number of top words per dimension shown to the judge (default: 5).")
    parser.add_argument("--dim-consistency-diversity", type=_parse_bool, dest="dim_consistency_diversity",
                        default=None,
                        help="Rescale the score by top-word diversity across dimensions (default: true).")
    parser.add_argument("--dim-consistency-model", type=str, dest="dim_consistency_model", default=None,
                        help="HF model id of the judge (default: Qwen/Qwen2.5-0.5B-Instruct).")
    parser.add_argument("--dim-consistency-method", type=str, dest="dim_consistency_method", default=None,
                        choices=["score", "similarity", "both"],
                        help="Which DimConsistencyJudge method(s) to run at each semantic check: "
                             "'score' (per-dimension outlier task), 'similarity' "
                             "(nearest-neighbour outlier task via score_similarity_consistency()), "
                             "or 'both'.")
    parser.add_argument("--remove-oov", type=_parse_bool, dest="remove_OOV", default=None)
    parser.add_argument("--time-iteration", type=_parse_bool, dest="time_iteration", default=None)
    parser.add_argument("--save-intermediate", type=_parse_bool, dest="save_intermediate", default=None)
    parser.add_argument("--pool-trim-every", type=int, dest="pool_trim_every", default=None,
                        help="Trim the CuPy memory pool every N iterations (per shard device). "
                             "Defaults to --sem-check-every. Reclaims out-of-pool cuBLAS workspace headroom.")
    parser.add_argument("--log-file", type=str, dest="log_file", default=None)

    parsed = parser.parse_args(args=argv)
    parsed_dict = vars(parsed)


    # Resolve "all" sentinel for shared_factors now that order is known
    if parsed_dict.get("shared_factors") == "all":
        n = parsed_dict.get("order") or default_exp.order
        parsed_dict["shared_factors"] = tuple(sorted((i, j) for i in range(n) for j in range(i + 1, n)))

    # Build new ExperimentConfig from defaults, overriding only provided values
    exp_kwargs = {}
    for field in ("dataset", "method", "order", "divergence", "dim", "name",
                  "random_state", "epsilon", "init", "normalize_factors",
                  "shared_factors", "subsample_frac", "max_nnz", "objective",
                  "decomposition", "cp_inner_iters", "cp_scooch_kappa"):
        v = parsed_dict.get(field, None)
        if v is not None:
            exp_kwargs[field] = v

    # rank needs special treatment
    if parsed_dict.get("rank") is not None:
        order = parsed_dict.get("order") or default_exp.order
        exp_kwargs["rank"] = _parse_rank(parsed_dict["rank"], n_modes=order)

    new_exp = replace(default_exp, **exp_kwargs) if exp_kwargs else default_exp

    # Training overrides
    train_kwargs = {}
    train_fields = (
        "n_iter_max",
        "tol",
        "warmup_steps",
        "patience",
        "verbose",
        "return_errors",
        "largedim",
        "checkpoint_saving_steps",
        "resume",
        "n_gpus",
        "gpu_id",
        "subsample_warmup",
        "max_cpu_frac",
        "tier1",
        "overwrite",
        "data_dir",
        "hpc",
    )
    # argparse used dashes -> underscores mapping; check each
    for f in train_fields:
        if f in parsed_dict and parsed_dict[f] is not None:
            train_kwargs[f] = parsed_dict[f]

    new_train = replace(default_train, **train_kwargs) if train_kwargs else default_train

    # Eval overrides
    eval_kwargs = {}
    eval_fields = (
        "rec_check_every",
        "rec_log_every",
        "sem_check_every",
        "sem_error_type",
        "sem_primary_key",
        "sem_softmax_temperature",
        "sem_fitness_target",
        "dim_consistency",
        "dim_consistency_words",
        "dim_consistency_diversity",
        "dim_consistency_model",
        "dim_consistency_method",
        "remove_OOV",
        "time_iteration",
        "save_intermediate",
        "pool_trim_every",
        "log_file",
    )
    for f in eval_fields:
        if f in parsed_dict and parsed_dict[f] is not None:
            eval_kwargs[f] = parsed_dict[f]

    new_eval = replace(default_eval, **eval_kwargs) if eval_kwargs else default_eval

    return RunConfig(exp=new_exp, train=new_train, eval=new_eval)


def hf_config_for_dataset(dataset: str) -> HFStreamConfig:
    """Map a dataset label to an HFStreamConfig. Supports 'path:config' shorthand."""
    if dataset in {"fineweb-en", "fineweb_en", "fineweb-english"}:
        return HFStreamConfig(
            path="HuggingFaceFW/fineweb",
            config="CC-MAIN-2025-26",
            split="train",
            text_column="text",
        )
    if ":" in dataset:
        path, cfg = dataset.split(":", 1)
        return HFStreamConfig(path=path.strip(), config=cfg.strip() or None)
    return HFStreamConfig(path=dataset, config=None)


def parse_vector_run_config(argv: Optional[List[str]] = None) -> VectorRunConfig:
    """
    Parse CLI args and return a VectorRunConfig built from defaults with overrides.

    Rules:
    - VectorExperimentConfig defaults come from config.py
    - HFStreamConfig is derived from --dataset (via hf_config_for_dataset)
      unless overridden by --hf-path/--hf-config/--hf-split/--hf-text-column.

    Args:
        argv: optional list of arguments (like sys.argv[1:]). If None, argparse
              will read from the actual command line.
    Returns:
        VectorRunConfig with overrides applied only for flags provided by the user.
    """
    default_exp = VectorExperimentConfig()

    parser = argparse.ArgumentParser(
        description="Build a VectorRunConfig from defaults and CLI overrides"
    )

    parser.add_argument("--type",
                        type=str,
                        default=None,
                        help="Frames or syntactic vector creation")

    # High-level selection: dataset label used for HF mapping + output grouping
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset label (used for HF mapping + output grouping). "
             "Example: fineweb-en or a HF path or path:config",
    )

    # VectorExperimentConfig overrides
    parser.add_argument("--output-dir", type=Path, dest="output_dir", default=None)
    parser.add_argument("--rows-per-flush", type=int, dest="rows_per_flush", default=None)
    parser.add_argument("--rows-per-part", type=int, dest="rows_per_part", default=None)

    parser.add_argument("--target-vectors", type=int, dest="target_vectors", default=None)
    parser.add_argument("--max-text-length", type=int, dest="max_text_length", default=None)

    parser.add_argument("--spacy-model", type=str, dest="spacy_model", default=None)
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=None)
    parser.add_argument("--cpu-frac", type=float, dest="cpu_frac", default=None)

    parser.add_argument("--log-every-s", type=float, dest="log_every_s", default=None)

    parser.add_argument(
        "--pad-sentences",
        type=_parse_bool,
        dest="pad_sentences",
        default=None,
        help="Pad each sentence with n-1 <s> tokens and one </s> before extracting "
             "n-grams (default: true). Padded output goes to a separate '-bos-eos' dir.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom label used in output paths instead of the dataset+config combination.",
    )

    # HFStreamConfig overrides (optional)
    parser.add_argument("--hf-path", type=str, dest="hf_path", default=None)
    parser.add_argument("--hf-config", type=str, dest="hf_config", default=None)
    parser.add_argument("--hf-split", type=str, dest="hf_split", default=None)
    parser.add_argument("--hf-text-column", type=str, dest="hf_text_column", default=None)

    parsed = parser.parse_args(args=argv)
    d = vars(parsed)

    # ---- exp overrides ----
    exp_kwargs = {}
    exp_fields = (
        "type",
        "output_dir",
        "rows_per_flush",
        "rows_per_part",
        "target_vectors",
        "max_text_length",
        "spacy_model",
        "batch_size",
        "cpu_frac",
        "log_every_s",
        "pad_sentences",
        "name",
    )
    for f in exp_fields:
        if d.get(f) is not None:
            exp_kwargs[f] = d[f]

    # Normalise n-gram type strings:
    #   "3-gram"         → "3gram"
    #   "4gram,5-gram"   → "4gram,5gram"
    #   "raw-3-gram"     → "raw3gram"
    #   "raw3gram,raw5gram" → "raw3gram,raw5gram"
    if "type" in exp_kwargs:
        raw_type = exp_kwargs["type"]
        parts = [p.strip() for p in raw_type.split(",") if p.strip()]
        normalised = []
        all_known = True
        for part in parts:
            n = parse_ngram_order(part)
            if n is not None:
                normalised.append(f"{n}gram")
                continue
            rn = parse_raw_ngram_order(part)
            if rn is not None:
                normalised.append(f"raw{rn}gram")
                continue
            all_known = False
            normalised.append(part)
        if all_known and normalised:
            exp_kwargs["type"] = ",".join(normalised)

    new_exp = replace(default_exp, **exp_kwargs) if exp_kwargs else default_exp

    # ---- hf config: derive from dataset unless overridden ----
    dataset = d.get("dataset") or "fineweb-en"
    hf_default = hf_config_for_dataset(dataset)

    hf_path = d.get("hf_path")
    hf_config = d.get("hf_config")
    hf_split = d.get("hf_split")
    hf_text_column = d.get("hf_text_column")

    # If user provides hf-path, it wins and we treat dataset as just a label.
    # If user does NOT provide hf-path, we use the derived default from dataset.
    hf = HFStreamConfig(
        path=hf_path if hf_path is not None else hf_default.path,
        config=hf_config if hf_config is not None else hf_default.config,
        split=hf_split if hf_split is not None else hf_default.split,
        text_column=hf_text_column if hf_text_column is not None else hf_default.text_column,
    )

    return VectorRunConfig(exp=new_exp, hf=hf)



def parse_population_run_config(argv: Optional[List[str]] = None) -> PopulationRunConfig:
    """Parse CLI args and return a PopulationRunConfig built from defaults with overrides."""
    default_exp = PopulationExperimentConfig()

    parser = argparse.ArgumentParser(description="Build a PopulationRunConfig from defaults and CLI overrides")

    parser.add_argument("--dataset", type=str, default=None, help="Dataset folder name inside vectors/ and tensors/")
    parser.add_argument("--top-ks", type=_parse_top_ks, default=None, help="Comma-separated ints, e.g. --top-ks 1000,2000,5000")
    parser.add_argument("--top-ks-asymmetric", type=_parse_top_ks_asymmetric, dest="top_ks_asymmetric", default=None,
                        help="Per-mode vocab sizes. Variants separated by '|', modes by ','. "
                             "E.g. --top-ks-asymmetric 1000,2000,1000|2000,4000,2000")
    # parser.add_argument("--v-col", type=str, dest="v_col", default=None)
    # parser.add_argument("--s-col", type=str, dest="s_col", default=None)
    # parser.add_argument("--o-col", type=str, dest="o_col", default=None)
    parser.add_argument(
        "--cols-to-build",
        type=_parse_cols_to_build,
        dest="cols_to_build",
        default=None,
        help='Comma-separated column names, e.g. --cols-to-build root,nsubj,obj or frame_name,target,arg1,arg2',
    )
    parser.add_argument(
        "--shared-factors",
        type=_parse_shared_factors,
        default=None,
        help="Factor linking for population, e.g. --shared-factors 2-3 or 1-2,2-3 or 'all'. Use 'none' to disable.",
    )
    parser.add_argument("--batch-rows", type=int, dest="batch_rows", default=None)
    parser.add_argument("--batch-readahead", type=int, dest="batch_readahead", default=None)
    parser.add_argument("--fragment-readahead", type=int, dest="fragment_readahead", default=None)
    parser.add_argument("--max-workers", type=int, dest="max_workers", default=None,
                        help="Max parallel worker processes for Pass 2. 0 or omit = auto "
                             "(scale to cores via --cpu-frac, capped by the memory ceiling).")
    parser.add_argument("--cpu-frac", type=float, dest="cpu_frac", default=None,
                        help="Fraction of CPU cores to target when --max-workers is auto. "
                             "Default 0.5 (polite locally); use 1.0 to fill a dedicated HPC node.")
    parser.add_argument("--max-mem-gb", type=float, dest="max_mem_gb", default=None,
                        help="Explicit RAM ceiling in GB used to cap the worker count so the run "
                             "does not swap. Auto-detected (SLURM/cgroup/psutil) if omitted.")
    parser.add_argument("--mem-per-worker-gb", type=float, dest="mem_per_worker_gb", default=None,
                        help="Override the per-worker RAM estimate (GB). Tune from observed RSS "
                             "if the shard-size-based estimate is off for your data.")
    parser.add_argument("--shards-per-task", type=int, dest="shards_per_task", default=None,
                        help="Shards bundled per submitted task. 1 = finest memory/ETA granularity (default).")
    parser.add_argument("--vectors-dir", type=Path, dest="vectors_dir_override", default=None,
                        help="Direct path to parquet vectors directory; overrides the dataset-derived path.")
    parser.add_argument("--data-dir", type=Path, dest="data_dir", default=None)
    parser.add_argument("--remove-hapax", type=_parse_bool, dest="remove_hapax", default=None,
                        help="Remove multi-way co-occurrences that appear only once before populating tensors.")
    parser.add_argument("--min-mode-ks", type=_parse_top_ks, dest="min_mode_ks", default=None,
                        help="Comma-separated per-mode minimum vocab floor, e.g. --min-mode-ks 5000,0,0 "
                             "guarantees at least 5K items from mode 0 in any shared vocabulary.")
    parser.add_argument("--ensured-vocab", type=_parse_ensured_vocab, dest="ensured_vocab", default=None,
                        help="Comma-separated token strings pinned into the shared vocabulary by name, "
                             "regardless of harmonic mean score. E.g. --ensured-vocab '<BOS>,<EOS>'. "
                             "Tokens absent from all marginals are silently skipped.")
    parser.add_argument("--tensors-to-build", type=_parse_cols_to_build, dest="tensors_to_build", default=None,
                        help="Comma-separated list of tensor names to build. Omit for the default set "
                             "(countingLog,countingLogEps,scSoftPlus). "
                             "Valid: counting,countingLog,countingLogEps,probLog,probLogSoftPlus,probLogShifted,sii,siiSoftPlus,siiShifted,sc,scSoftPlus,scShifted,scSoftPlusFlat. "
                             "E.g. --tensors-to-build counting,countingLog,sc")

    parsed = parser.parse_args(args=argv)
    d = vars(parsed)

    # If only asymmetric variants were specified, don't pull in the default uniform top_ks.
    if d.get("top_ks_asymmetric") is not None and d.get("top_ks") is None:
        d["top_ks"] = ()

    # Resolve "all" sentinel for shared_factors using cols_to_build length
    if d.get("shared_factors") == "all":
        cols = d.get("cols_to_build") or default_exp.cols_to_build
        n = len(cols)
        d["shared_factors"] = tuple(sorted((i, j) for i in range(n) for j in range(i + 1, n)))

    exp_kwargs = {}
    for f in (
            "dataset",
            "top_ks",
            "top_ks_asymmetric",
            "cols_to_build",
            "shared_factors",
            "batch_rows",
            "batch_readahead",
            "fragment_readahead",
            "max_workers",
            "cpu_frac",
            "max_mem_gb",
            "mem_per_worker_gb",
            "shards_per_task",
            "vectors_dir_override",
            "data_dir",
            "remove_hapax",
            "min_mode_ks",
            "ensured_vocab",
            "tensors_to_build",
    ):
        if d.get(f) is not None:
            exp_kwargs[f] = d[f]

    new_exp = replace(default_exp, **exp_kwargs) if exp_kwargs else default_exp
    return PopulationRunConfig(exp=new_exp)


if __name__ == "__main__":
    # Keep your existing behavior for parse_run_config() when run directly
    # cfg = parse_run_config()
    cfg = parse_population_run_config()
    print(json.dumps(asdict(cfg), default=str, indent=2))
