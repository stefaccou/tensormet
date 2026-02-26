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
    for field in ("dataset", "method", "divergence", "dim", "name", "random_state", "max_cpu_frac", "data_dir"):
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


if __name__ == "__main__":
    # When executed directly, parse sys.argv and print resulting config as JSON
    cfg = parse_run_config()
    print(json.dumps(asdict(cfg), default=str, indent=2))

