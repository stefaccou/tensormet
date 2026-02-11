from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
import hashlib
import json
import platform
import time
from datetime import datetime

@dataclass(frozen=True)
class TrainingConfig:
    n_iter_max: int = 1000
    tol: float = 1e-5
    epsilon: float = 1e-12
    init: str = "random"
    normalize_factors: bool = False
    warmup_steps: int = 1
    patience: int = 5
    verbose: bool = True
    return_errors: str = "full"

@dataclass(frozen=True)
class EvalConfig:
    rec_check_every: int = 0
    rec_log_every: int = rec_check_every # if not passed, we log at any rec_check step
    sem_check_every: int = 20
    sem_error_type: str = "average_rank_score"
    sem_softmax_temperature: float = 0.1
    sem_fitness_target: int = 10_000
    n_sentence_cache: Optional[int] = None  # if we later want to cap loaded sentences
    remove_OOV: bool = False # whether to set OOV in test set to OOV token (false ignores the sentences)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    method: str
    divergence: str
    dim: int
    rank: Tuple[int, ...]
    name: str
    random_state: int = 1
    max_cpu_frac: float = 0.5

    # paths
    data_dir: Path = Path(".")

    def run_id(self) -> str:
        """Stable-ish identifier based on config content (not timestamp)."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:10]

    def output_dir(self) -> Path:
        return self.data_dir / "tensors" / self.dataset / "decomposition"

    def model_filename(self, train: TrainingConfig) -> str:
        # mirrors current naming but makes it deterministic
        r0 = self.rank[0] if len(self.rank) > 0 else "r"
        return (f"{self.name + '_' if self.name else ''}"
                f"{self.divergence}_{self.method}_{self.dim}d_{r0}r_{train.n_iter_max}i.pt")

    def output_paths(self, train: TrainingConfig) -> Dict[str, Path]:
        out_dir = self.output_dir()
        model_path = out_dir / self.model_filename(train)
        return {
            "model": model_path,
            "errors": model_path.with_name(model_path.stem + "_errors.npy"),
            "fitness": model_path.with_name(model_path.stem + "_fitness.npy"),
            "config": model_path.with_name(model_path.stem + "_config.json"),
            "log_jsonl": out_dir / "runs.jsonl",
        }
@dataclass(frozen=True)
class RunConfig:
    exp: ExperimentConfig
    train: TrainingConfig
    eval: EvalConfig

    def run_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:10]

    def output_dir(self) -> Path:
        return self.exp.data_dir / "tensors" / self.exp.dataset / "decomposition"

    def model_path(self) -> Path:
        r0 = self.exp.rank[0] if len(self.exp.rank) else "r"
        fname = f"{self.exp.name + '_' if self.exp.name else ''}{self.exp.divergence}_{self.exp.method}_{self.exp.dim}d_{r0}r_{self.train.n_iter_max}i.pt"
        return self.output_dir() / fname

    def artifact_paths(self) -> Dict[str, Path]:
        model = self.model_path()
        return {
            "model": model,
            "errors": model.with_name(model.stem + "_errors.npy"),
            "fitness": model.with_name(model.stem + "_fitness.npy"),
            "config":model.with_name(model.stem + "_config.json"),
            "runs_jsonl": self.output_dir() / "runs.jsonl",
        }
