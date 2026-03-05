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
    _default_hf_config_for_dataset #todo Fix this import
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


def _parse_rank(s: str, n_modes=3) -> Tuple[int, ...]:
    # Accept comma-separated integers like "100,100,100" or a single int
    if not s:
        return tuple()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < n_modes:
        parts = [parts[0]]*n_modes
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid rank specification: {s}")

def _parse_shared_factors(s: str):
    """
    Parse shared factor links.

    Accepts:
      --shared-factors none
      --shared-factors 1-2
      --shared-factors 1-2,2-0
      --shared-factors 1:2,2:0

    Returns:
      None  (if 'none'/'null'/'')
      set({(a,b), ...})
    """
    if s is None:
        return None
    s2 = str(s).strip().lower()
    if s2 in ("", "none", "null", "no"):
        return None

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
        pairs.add((ai, bi))

    return pairs if pairs else None


def _none_if_missing(value, sentinel=None):
    # Helper: treat argparse's default sentinel as missing -> return None
    return None if value is sentinel else value


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
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--rank", type=_parse_rank, default=None,
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
        help="Factor linking, e.g. --shared-factors 1-2,2-0 . Use 'none' to disable.",
    )
    parser.add_argument("--warmup-steps", type=int, dest="warmup_steps", default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--verbose", type=_parse_bool, default=None)
    parser.add_argument("--return-errors", type=str, dest="return_errors", default=None)
    parser.add_argument("--largedim", type=_parse_bool, default=None)
    parser.add_argument("--checkpoint-saving-steps", type=int, dest="checkpoint_saving_steps", default=None)

    # Eval-level args
    parser.add_argument("--rec-check-every", type=int, dest="rec_check_every", default=None)
    parser.add_argument("--rec-log-every", type=int, dest="rec_log_every", default=None)
    parser.add_argument("--sem-check-every", type=int, dest="sem_check_every", default=None)
    parser.add_argument("--sem-error-type", type=str, dest="sem_error_type", default=None)
    parser.add_argument("--sem-softmax-temperature", type=float, dest="sem_softmax_temperature", default=None)
    parser.add_argument("--sem-fitness-target", type=int, dest="sem_fitness_target", default=None)
    parser.add_argument("--n-sentence-cache", type=int, dest="n_sentence_cache", default=None)
    parser.add_argument("--remove-oov", type=_parse_bool, dest="remove_OOV", default=None)
    parser.add_argument("--time-iteration", type=_parse_bool, dest="time_iteration", default=None)
    parser.add_argument("--save-intermediate", type=_parse_bool, dest="save_intermediate", default=None)
    parser.add_argument("--log-file", type=str, dest="log_file", default=None)

    parsed = parser.parse_args(args=argv)
    parsed_dict = vars(parsed)

    # Build new ExperimentConfig from defaults, overriding only provided values
    exp_kwargs = {}
    for field in ("dataset", "method", "divergence", "dim", "name",
                  "random_state", "max_cpu_frac", "data_dir", "overwrite", "tier1"):
        v = parsed_dict.get(field, None)
        if v is not None:
            exp_kwargs[field] = v

    # rank needs special treatment
    if parsed_dict.get("rank") is not None:
        exp_kwargs["rank"] = parsed_dict["rank"]

    new_exp = replace(default_exp, **exp_kwargs) if exp_kwargs else default_exp

    # Training overrides
    train_kwargs = {}
    train_fields = (
        "n_iter_max",
        "tol",
        "epsilon",
        "init",
        "normalize_factors",
        "shared_factors",
        "warmup_steps",
        "patience",
        "verbose",
        "return_errors",
        "largedim",
        "checkpoint_saving_steps",
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
        "sem_softmax_temperature",
        "sem_fitness_target",
        "n_sentence_cache",
        "remove_OOV",
        "time_iteration",
        "save_intermediate",
        "log_file",
    )
    for f in eval_fields:
        if f in parsed_dict and parsed_dict[f] is not None:
            eval_kwargs[f] = parsed_dict[f]

    new_eval = replace(default_eval, **eval_kwargs) if eval_kwargs else default_eval

    return RunConfig(exp=new_exp, train=new_train, eval=new_eval)


def parse_vector_run_config(argv: Optional[List[str]] = None) -> VectorRunConfig:
    """
    Parse CLI args and return a VectorRunConfig built from defaults with overrides.

    Rules:
    - VectorExperimentConfig defaults come from config.py
    - HFStreamConfig is derived from --dataset (via _default_hf_config_for_dataset)
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
        "output_dir",
        "rows_per_flush",
        "rows_per_part",
        "target_vectors",
        "max_text_length",
        "spacy_model",
        "batch_size",
        "cpu_frac",
        "log_every_s",
    )
    for f in exp_fields:
        if d.get(f) is not None:
            exp_kwargs[f] = d[f]
    new_exp = replace(default_exp, **exp_kwargs) if exp_kwargs else default_exp

    # ---- hf config: derive from dataset unless overridden ----
    dataset = d.get("dataset") or "fineweb-en"
    hf_default = _default_hf_config_for_dataset(dataset)

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


if __name__ == "__main__":
    # Keep your existing behavior for parse_run_config() when run directly
    # cfg = parse_run_config()
    cfg = parse_vector_run_config()
    print(json.dumps(asdict(cfg), default=str, indent=2))
