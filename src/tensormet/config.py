from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional, Dict, Union, Set, Any
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
    shared_factors: Optional[Tuple[Tuple[int, int], ...]] = None
    warmup_steps: int = 1
    patience: int = 5
    verbose: bool = True
    return_errors: str = "full"
    largedim: bool = False
    checkpoint_saving_steps: int = 0 # defaults to 0 -> Falsy
    resume: bool = False  # <-- ADDED for checkpoint resumption

@dataclass(frozen=True)
class EvalConfig:
    rec_check_every: int = 20
    rec_log_every: int = 20 # defaults to rec_check_every if not passed
    sem_check_every: int = 20
    sem_error_type: Union[str, Tuple[str, ...]] = "full" # updated 2026-03-04
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
    order: int=3
    rank: Tuple[int, ...] = (100, 100, 100)
    name: str = None
    random_state: int = 1
    max_cpu_frac: float = 1
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
        return (f"{prefix}{self.exp.divergence}_{self.exp.method}_{self.exp.order}D_"
                f"{self.exp.dim}d_{r0}r_{self.train.n_iter_max}i.pt")

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

    def get_resume_state(self) -> Dict[str, Any]:
        """
        Parses the artifact directory to find the latest checkpoint and historical metrics
        from ANY compatible run (matching structural hyperparameters, ignoring iteration counts).
        Returns a kwargs dict ready to be unpacked into `non_negative_tucker_with_similarity`.
        """
        if not self.train.resume:
            return {}

        paths = self.artifact_paths()
        out_dir = paths["model"].parent

        # 1. Build wildcard patterns for the base name.
        # Try new naming first (includes {order}D_), then fall back to legacy (no order prefix).
        r0 = self.exp.rank[0] if len(self.exp.rank) else "r"
        prefix = f"{self.exp.name}_" if self.exp.name else ""
        new_pattern = f"{prefix}{self.exp.divergence}_{self.exp.method}_{self.exp.order}D_{self.exp.dim}d_{r0}r_"
        legacy_pattern = f"{prefix}{self.exp.divergence}_{self.exp.method}_{self.exp.dim}d_{r0}r_"

        # Find all JSON config files matching either pattern
        candidate_configs = list(out_dir.glob(f"{new_pattern}*i_config.json"))
        if not candidate_configs:
            candidate_configs = list(out_dir.glob(f"{legacy_pattern}*i_config.json"))
            if candidate_configs:
                print(f"No new-style ({self.exp.order}D) checkpoints found; falling back to legacy naming.")

        best_candidate_paths = paths
        latest_iter = -1

        def _canonical_shared_factors(x):
            if x is None:
                return None
            return tuple(sorted(tuple(sorted(pair)) for pair in x))

        # 2. Iterate through found configs and verify structural compatibility
        for config_path in candidate_configs:
            print("investigating", config_path)
            try:
                with open(config_path, "r") as f:
                    old_cfg_data = json.load(f).get("cfg", {})
            except Exception:
                continue

            old_exp = old_cfg_data.get("exp", {})
            old_train = old_cfg_data.get("train", {})


            # These are the variables that MUST match to safely resume
            is_compatible = (
                    old_exp.get("dataset") == self.exp.dataset and
                    old_exp.get("order") == self.exp.order and # new: order
                    old_exp.get("method") == self.exp.method and
                    old_exp.get("divergence") == self.exp.divergence and
                    old_exp.get("dim") == self.exp.dim and
                    tuple(old_exp.get("rank", [])) == tuple(self.exp.rank) and
                    old_train.get("init") == self.train.init and
                    _canonical_shared_factors(old_train.get("shared_factors")) ==
                    _canonical_shared_factors(self.train.shared_factors)
            )
            print(old_exp.get("dataset"), self.exp.dataset, "\n",
            old_exp.get("method"),self.exp.method, "\n",
            old_exp.get("order"),self.exp.order, "\n", #new: order
            old_exp.get("divergence"), self.exp.divergence, "\n",
            old_exp.get("dim"), self.exp.dim, "\n",
            tuple(old_exp.get("rank", [])), tuple(self.exp.rank), "\n",
            old_train.get("init"), self.train.init, "\n",
            # Cast both to string to safely compare parsed JSON strings with Python sets
            _canonical_shared_factors(old_train.get("shared_factors")),
            _canonical_shared_factors(self.train.shared_factors))


            if is_compatible:
                stem = config_path.name.replace("_config.json", "")
                candidate_ckpt_dir = out_dir / f"{stem}_checkpoints"

                if candidate_ckpt_dir.exists():
                    pt_files = list(candidate_ckpt_dir.glob("*.pt"))
                    iterations = [int(p.stem) for p in pt_files if p.stem.isdigit()]
                    if iterations:
                        max_i = max(iterations)
                        # We want the run that progressed the furthest
                        if max_i > latest_iter:
                            print(candidate_ckpt_dir, "gives new best with", max_i)
                            latest_iter = max_i
                            # Map the paths to the old run so we load its history perfectly
                            best_candidate_paths = {
                                "errors": out_dir / f"{stem}_errors.npy",
                                "fitness": out_dir / f"{stem}_fitness.npy",
                                "fitness_json": out_dir / f"{stem}_fitness.json",
                                "checkpoint_dir": candidate_ckpt_dir,
                            }

        ckpt_dir = best_candidate_paths.get("checkpoint_dir")

        if latest_iter == -1 or not ckpt_dir or not ckpt_dir.exists():
            print(f"Warning: Resume flag is True, but no compatible checkpoints found. Starting from scratch.")
            return {}

        ckpt_path = ckpt_dir / f"{latest_iter}.pt"

        # Local imports
        import numpy as np
        import torch

        # 3. Load reconstruction errors
        rec_errors = []
        if best_candidate_paths.get("errors") and best_candidate_paths["errors"].exists():
            rec_errors = np.load(best_candidate_paths["errors"]).tolist()
            # truncate to latest_iter just in case a crash happened mid-save
            rec_errors = rec_errors[:latest_iter]

        # 4. Load fitness scores
        fitness_scores = []
        if best_candidate_paths.get("fitness_json") and best_candidate_paths["fitness_json"].exists():
            with open(best_candidate_paths["fitness_json"], "r") as f:
                fitness_scores = json.load(f)
        elif best_candidate_paths.get("fitness") and best_candidate_paths["fitness"].exists():
            fitness_scores = np.load(best_candidate_paths["fitness"]).tolist()

        # 5. Reconstruct the best semantic score
        best_sem_score = 0.0
        if fitness_scores:
            sem_key = self.eval.sem_error_type
            if isinstance(sem_key, (list, tuple)):
                sem_key = sem_key[0]

            for score in fitness_scores:
                if isinstance(score, dict):
                    if sem_key in score:
                        val = float(score[sem_key])
                    elif sem_key == "all" and "average_rank_score" in score:
                        # Fallback to the default metric if "all" was used
                        val = float(score["average_rank_score"])
                    else:
                        # Fallback to the first available metric if the key is missing
                        try:
                            val = float(list(score.values())[0])
                        except (ValueError, TypeError, IndexError):
                            continue
                else:
                    try:
                        val = float(score)
                    except (ValueError, TypeError):
                        continue

                if val > best_sem_score:
                    best_sem_score = val

        # 6. Load the model weights
        checkpoint_tensor = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        print(
            f"Resuming from compatible run! Loaded iteration {latest_iter} with best semantic score {best_sem_score:.4f}")

        return {
            "start_iteration": latest_iter,
            "best_sem_score": best_sem_score,
            "rec_errors": rec_errors,
            "fitness_scores": fitness_scores,
            "checkpoint_tensor": checkpoint_tensor
        }
@dataclass(frozen=True)
class VectorExperimentConfig:

    type: str = "syntactic"
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


@dataclass(frozen=True)
class PopulationExperimentConfig:
    dataset: str = "fineweb-en"
    top_ks: Tuple[int, ...] = (1000, 2000, 4000, 6000)
    cols_to_build: Tuple[str, ...] = ("root", "nsubj", "obj")
    shared_factors: Optional[Tuple[Tuple[int, int], ...]] = None
    # v_col: str = "root"
    # s_col: str = "nsubj"
    # o_col: str = "obj"
    batch_rows: int = 256_000
    batch_readahead: int = 32
    fragment_readahead: int = 8
    data_dir: Path = DATA_DIR

    def vectors_dir(self) -> Path:
        return self.data_dir / "vectors" / self.dataset

    def output_dir(self) -> Path:
        return self.data_dir / "tensors" / self.dataset

@dataclass(frozen=True)
class PopulationRunConfig:
    exp: PopulationExperimentConfig