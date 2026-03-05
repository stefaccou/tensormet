from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional, Dict, Union, Set
import hashlib
import json
from tensormet.utils import DATA_DIR

@dataclass(frozen=True)
class TrainingConfig:
    n_iter_max: int = 1000
    tol: float = 1e-5
    epsilon: float = 1e-12
    init: str = "random"
    normalize_factors: bool = False
    shared_factors: Optional[Set[Tuple[int, int]]] = None
    warmup_steps: int = 1
    patience: int = 5
    verbose: bool = True
    return_errors: str = "full"
    largedim: bool = False
    checkpoint_saving_steps: int = 0 # defaults to 0 -> Falsy

@dataclass(frozen=True)
class EvalConfig:
    rec_check_every: int = 20
    rec_log_every: int = rec_check_every # if not passed, we log at any rec_check step
    sem_check_every: int = 20
    sem_error_type: Union[str, Tuple[str, ...]] = "average_rank_score" # updated 2026-03-04
    sem_softmax_temperature: float = 0.1
    sem_fitness_target: int = 10_000
    n_sentence_cache: Optional[int] = None  # if we later want to cap loaded sentences
    remove_OOV: bool = False # whether to set OOV in test set to OOV token (false ignores the sentences)
    time_iteration: bool = True # whether to print the time taken by an iteration
    save_intermediate: bool = True # whether to save the current best model (safety for interrupted code)
    log_file: Optional[Union[str, Path]] = None


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "fineweb-en"
    method: str = "siiSoftPlus"
    divergence: str = "fr"
    dim: int = 1000
    rank: Tuple[int, ...] = (100, 100, 100)
    name: str = None
    random_state: int = 1
    max_cpu_frac: float = 0.5
    tier1: bool = False
    overwrite: bool = False
    # paths
    data_dir: Path = DATA_DIR

    def run_id(self) -> str:
        """Stable-ish identifier based on config content (not timestamp)."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:10]

    def output_dir(self) -> Path:
        return self.data_dir / "tensors" / self.dataset / "decomposition"

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

    def model_filename(self) -> str:
        # deterministic + readable
        r0 = self.exp.rank[0] if len(self.exp.rank) else "r"
        prefix = f"{self.exp.name}_" if self.exp.name else ""
        return f"{prefix}{self.exp.divergence}_{self.exp.method}_{self.exp.dim}d_{r0}r_{self.train.n_iter_max}i.pt"

    def model_path(self) -> Path:
        return self.output_dir() / self.model_filename()

    def artifact_paths(self) -> Dict[str, Path]:
        """
        Canonical artifact paths for this run.
        Keep everything derived from model_path() so downstream code never re-invents paths.
        """
        model = self.model_path()
        out_dir = model.parent

        checkpoint_dir = out_dir / f"{model.stem}_checkpoints"

        # allow overriding log path from EvalConfig, otherwise default next to model
        if self.eval.log_file is not None:
            log_path = Path(self.eval.log_file)
            if not log_path.is_absolute():
                log_path = out_dir / log_path
        else:
            log_path = model.with_name(model.stem + "_log.txt")

        return {
            "model": model,
            "errors": model.with_name(model.stem + "_errors.npy"),

            # keep the old npy path for scalar semantics,
            # and add json for dict semantics (multi-key / all)
            "fitness": model.with_name(model.stem + "_fitness.npy"),
            "fitness_json": model.with_name(model.stem + "_fitness.json"),

            "config": model.with_name(model.stem + "_config.json"),
            "runs_jsonl": out_dir / "runs.jsonl",
            "log": log_path,
            "checkpoint_dir": checkpoint_dir,
        }

@dataclass(frozen=True)
class VectorExperimentConfig:
    # Output + resume
    output_dir: Path = DATA_DIR / "vectors"
    rows_per_flush: int = 100_000
    rows_per_part: int = 5_000_000

    # Stream controls
    target_vectors: int = 10_000_000
    max_text_length: int = 50_000

    # spaCy controls
    spacy_model: str = "en_core_web_md"
    batch_size: int = 256
    cpu_frac: float = 0.66

    # logging
    log_every_s: float = 30.0

@dataclass(frozen=True)
class HFStreamConfig:
    """How to stream texts from a HF dataset."""
    path: str
    config: Optional[str]
    split: str = "train"
    text_column: str = "text"

def _default_hf_config_for_dataset(dataset: str) -> HFStreamConfig:
    """
    Map your ExperimentConfig.dataset to a HF streaming spec.
    Keep this small + explicit so downstream code stays consistent.
    """
    # Your current default in ExperimentConfig is "fineweb-en".
    if dataset in {"fineweb-en", "fineweb_en", "fineweb-english"}:
        return HFStreamConfig(
            path="HuggingFaceFW/fineweb",
            config="CC-MAIN-2025-26",
            split="train",
            text_column="text",
        )

    # Fallback: allow passing a HF dataset path directly in cfg.exp.dataset
    # Optionally support "path:config" form.
    if ":" in dataset:
        path, cfg = dataset.split(":", 1)
        cfg = cfg.strip() or None
        return HFStreamConfig(path=path.strip(), config=cfg)

    return HFStreamConfig(path=dataset, config=None)

@dataclass(frozen=True)
class VectorRunConfig:
    exp: VectorExperimentConfig
    hf: HFStreamConfig

    def output_dir(self) -> Path:
        dataset = self.hf.path.replace("/", "-").strip()
        config = (self.hf.config or "").replace("/", "-").strip()
        return self.exp.output_dir / f"{dataset}_{config}_{self.exp.target_vectors}v"