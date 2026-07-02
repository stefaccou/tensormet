from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Tuple, Optional, Dict, Union, Set, Any
import hashlib
import json
import re
import os
from tensormet.utils import DATA_DIR, shared_factor_suffix, nontrivial_linked_groups, dim_spec_str
from tensormet.naming import (
    model_filename as _model_filename,
    candidate_stems as _candidate_stems,
    vocab_filename as _vocab_filename,
)


def _as_dim_tuple(dim) -> tuple:
    """Normalise dim to a tuple regardless of whether it is stored as int or list/tuple."""
    if isinstance(dim, int):
        return (dim,)
    return tuple(dim)


def parse_ngram_order(type_str: str) -> Optional[int]:
    """Parse a single n-gram type string like '3gram', '3-gram' → n (int).

    Returns None if the string is not a recognised n-gram pattern.
    Does not handle comma-separated multi-gram strings; use parse_ngram_orders for that.
    """
    m = re.fullmatch(r"(\d+)-?gram", type_str.strip().lower())
    return int(m.group(1)) if m else None


def parse_ngram_orders(type_str: str) -> Optional[list]:
    """Parse single or comma-separated n-gram type strings → sorted list of ints.

    Examples:
        "3gram"         → [3]
        "4-gram"        → [4]
        "4gram,5gram"   → [4, 5]
        "3-gram,4gram,5-gram" → [3, 4, 5]
        "syntactic"     → None
    """
    orders = []
    for part in type_str.split(","):
        n = parse_ngram_order(part.strip())
        if n is None:
            return None   # any non-ngram token → not an ngram type string
        orders.append(n)
    return sorted(set(orders)) if orders else None


def parse_raw_ngram_order(type_str: str) -> Optional[int]:
    """Parse a single raw n-gram type string like 'raw3gram', 'raw-3-gram' → n (int).

    Returns None if the string is not a recognised raw-ngram pattern.
    """
    m = re.fullmatch(r"raw-?(\d+)-?gram", type_str.strip().lower())
    return int(m.group(1)) if m else None


def parse_raw_ngram_orders(type_str: str) -> Optional[list]:
    """Parse single or comma-separated raw-ngram type strings → sorted list of ints.

    Examples:
        "raw3gram"           → [3]
        "raw-4-gram"         → [4]
        "raw3gram,raw5gram"  → [3, 5]
        "3gram"              → None  (not a raw-ngram string)
    """
    orders = []
    for part in type_str.split(","):
        n = parse_raw_ngram_order(part.strip())
        if n is None:
            return None
        orders.append(n)
    return sorted(set(orders)) if orders else None

@dataclass(frozen=True)
class TrainingConfig:
    n_iter_max: int = 1000
    tol: float = 1e-5
    warmup_steps: int = 1
    patience: int = 5
    verbose: bool = True
    return_errors: str = "full"
    largedim: bool = False
    checkpoint_saving_steps: int = 0
    resume: bool = False
    n_gpus: int = 1
    gpu_id: Optional[Union[int, Tuple[int, ...]]] = None
    subsample_warmup: int = 0
    max_cpu_frac: float = 1
    tier1: bool = False
    overwrite: bool = False
    data_dir: Path = DATA_DIR
    # HPC mode: stage all per-run artifact writes (model, errors, fitness,
    # checkpoints, log) to node-local scratch ($TMPDIR) during the run, then
    # copy them back to data_dir (shared GPFS) once at the end. Relieves the
    # metadata-lock contention that explodes iteration times when many array
    # tasks write to the same GPFS decomposition directory simultaneously.
    hpc: bool = False

@dataclass(frozen=True)
class EvalConfig:
    rec_check_every: int = 20
    rec_log_every: Optional[int] = None  # None/0 → falls back to rec_check_every
    sem_check_every: int = 20
    sem_error_type: Union[str, Tuple[str, ...]] = "full" # updated 2026-03-04
    sem_primary_key: Optional[str] = None  # overrides auto-derived primary key for patience/diff/logging
    sem_softmax_temperature: float = 0.1
    sem_fitness_target: int = 10_000
    # LLM-as-judge "dimension consistency" scoring (see tensormet.judge). Default
    # OFF: the judge model costs ~1 GB of GPU memory on top of the decomposition.
    dim_consistency: bool = False
    dim_consistency_words: int = 5          # top words per dimension shown to the judge
    dim_consistency_diversity: bool = True  # rescale by distinct-top-word diversity
    dim_consistency_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    remove_OOV: bool = False # whether to set OOV in test set to OOV token (false ignores the sentences)
    time_iteration: bool = True # whether to print the time taken by an iteration
    save_intermediate: bool = True # whether to save the current best model (safety for interrupted code)
    # Proactive GPU pool trim cadence (iterations). None -> default to sem_check_every.
    # Every this-many iterations the CuPy memory pool's cached-but-unused blocks are
    # returned to the driver on each shard device, reclaiming transient eval/copy
    # memory so out-of-pool cuBLAS/cuSPARSE workspaces keep their headroom (the
    # device-0 CUBLASError cascade). Kept off the per-iteration hot path on purpose.
    pool_trim_every: Optional[int] = None
    log_file: Optional[Union[str, Path]] = None


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "fineweb-en"
    method: str = "siiSoftPlus"
    divergence: str = "fr"
    dim: Union[int, Tuple[int, ...]] = 1000
    order: int = 3
    rank: Tuple[int, ...] = (100, 100, 100)
    name: str = None
    random_state: int = 1
    epsilon: float = 1e-12
    init: str = "random"
    normalize_factors: bool = False
    shared_factors: Optional[Tuple[Tuple[int, int], ...]] = None
    subsample_frac: float = 1.0
    # "full"   -> fit the entire (zero-filled) tensor; correct for count/co-occurrence data
    #             where an unobserved entry genuinely means 0.
    # "masked" -> fit ONLY observed (nonzero) entries; treat the rest as missing
    #             (weighted/completion objective). Correct for recommendation/generalisation
    #             data (e.g. Netflix) where "unobserved" != "rated 0".
    objective: str = "full"

@dataclass(frozen=True)
class RunConfig:
    exp: ExperimentConfig
    train: TrainingConfig
    eval: EvalConfig

    def run_id(self) -> str:
        """Fingerprint of the FULL invocation (every exp/train/eval field).

        This is deliberately environment-inclusive: two runs that differ only in
        execution context (``gpu_id``, ``n_gpus``, ``data_dir``, ``hpc``,
        ``log_file``, …) get distinct ids. That is required by ``staging_root``,
        which uses ``run_id`` to isolate concurrent array tasks that happen to
        share a ``$TMPDIR`` — narrowing it to "scientific" fields would let two
        such tasks collide. It also keeps ``run_id`` recomputable from the ``cfg``
        stored alongside it in ``runs.jsonl`` / ``*_config.json`` (a 1:1 mapping
        back to the exact invocation).

        It is NOT the scientific/structural identity of an experiment. Artifact
        retrieval is keyed by ``model_filename()`` (divergence/method/order/dim/
        rank/iters/…), and resume-compatibility is decided by
        ``get_resume_state()`` comparing structural fields — neither uses run_id.
        """
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:10]

    def staging_root(self) -> Path:
        """Node-local scratch root used when ``train.hpc`` is set.

        Prefers ``$TMPDIR`` (per-job node-local SSD on most schedulers, incl.
        dodrio/Slurm), falling back to the OS temp dir. Runs are isolated by
        ``run_id`` so concurrent tasks that happen to share a $TMPDIR never
        collide.
        """
        import tempfile
        base = os.environ.get("TMPDIR") or tempfile.gettempdir()
        return Path(base) / "tensormet_stage" / self.run_id()

    def output_dir(self, *, staged: bool = False) -> Path:
        if staged:
            return self.staging_root() / "tensors" / self.exp.dataset / "decomposition"
        return self.train.data_dir / "tensors" / self.exp.dataset / "decomposition"

    def model_filename(self) -> str:
        # deterministic + readable
        return _model_filename(
            self.exp.divergence, self.exp.method, self.exp.order,
            self.exp.dim, self.exp.rank, self.train.n_iter_max,
            name=self.exp.name,
            shared_factors=self.exp.shared_factors,
            subsample_frac=self.exp.subsample_frac,
        )

    def model_path(self) -> Path:
        return self.output_dir() / self.model_filename()

    def artifact_paths(self, *, staged: Optional[bool] = None) -> Dict[str, Path]:
        """
        Artifact paths for this run.

        By default (``staged=None``) the location follows ``train.hpc``: when HPC
        mode is on, paths resolve under node-local ``$TMPDIR`` so the hot-loop
        writes (temp model, errors, fitness, checkpoints, log) never touch shared
        GPFS. Pass ``staged=False`` to force the canonical GPFS destination
        (e.g. for resume scans and the final copy-back target).

        Keep everything derived from the resolved out_dir so downstream code
        never re-invents paths.
        """
        if staged is None:
            staged = self.train.hpc
        out_dir = self.output_dir(staged=staged)
        model = out_dir / self.model_filename()

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

            "timing_json": model.with_name(model.stem + "_timing.json"),

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
        if not self.train.resume:  # resume stays in TrainingConfig
            return {}

        # Resume always reads prior artifacts from the canonical GPFS location;
        # under HPC mode the staged ($TMPDIR) tree is empty at job start.
        paths = self.artifact_paths(staged=False)
        out_dir = paths["model"].parent

        # 1. Build wildcard patterns for the base name.
        # Try new naming first (includes {order}D_), then fall back to legacy (no order prefix).
        stems = _candidate_stems(
            self.exp.divergence, self.exp.method, self.exp.order,
            self.exp.dim, self.exp.rank,
            name=self.exp.name,
            shared_factors=self.exp.shared_factors,
            subsample_frac=self.exp.subsample_frac,
        )

        # Find all JSON config files: new (with sf) → new (without sf) → legacy
        candidate_configs = []
        for stem in stems:
            candidate_configs = list(out_dir.glob(f"{stem}*i_config.json"))
            if candidate_configs:
                if stem == stems[-1] and len(stems) > 1:
                    print(f"No new-style ({self.exp.order}D) checkpoints found; falling back to legacy naming.")
                elif stem != stems[0]:
                    print(f"No shared-factor checkpoints found; falling back to non-shared naming.")
                break

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


            # These are the variables that MUST match to safely resume.
            # random_state is included because stochastic windows/sampling are a
            # pure function of (random_state, iteration): resuming across runs with
            # different seeds would splice together incompatible RNG streams.
            # Old configs predating this field fall back to the dataclass default.
            is_compatible = (
                    old_exp.get("dataset") == self.exp.dataset and
                    old_exp.get("order") == self.exp.order and
                    old_exp.get("method") == self.exp.method and
                    old_exp.get("divergence") == self.exp.divergence and
                    _as_dim_tuple(old_exp.get("dim", [])) == _as_dim_tuple(self.exp.dim) and
                    tuple(old_exp.get("rank", [])) == tuple(self.exp.rank) and
                    old_exp.get("init") == self.exp.init and
                    int(old_exp.get("random_state", 1)) == int(self.exp.random_state) and
                    _canonical_shared_factors(old_exp.get("shared_factors")) ==
                    _canonical_shared_factors(self.exp.shared_factors) and
                    float(old_exp.get("subsample_frac", 1.0)) == self.exp.subsample_frac
            )

            print(old_exp.get("dataset"), self.exp.dataset, "\t",
            old_exp.get("method"), self.exp.method, "\t",
            old_exp.get("order"), self.exp.order, "\t",
            old_exp.get("divergence"), self.exp.divergence, "\t",
            old_exp.get("dim"), self.exp.dim, "\t",
            tuple(old_exp.get("rank", [])), tuple(self.exp.rank), "\t",
            old_exp.get("init"), self.exp.init, "\t",
            old_exp.get("random_state", 1), self.exp.random_state, "\t",
            _canonical_shared_factors(old_exp.get("shared_factors")),
            _canonical_shared_factors(self.exp.shared_factors), "\t",
            float(old_exp.get("subsample_frac", 1.0)), self.exp.subsample_frac)


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
            if self.eval.sem_primary_key is not None:
                sem_key = self.eval.sem_primary_key
            else:
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


@dataclass
class InspectionConfig:
    dim: Union[int, str, Tuple[int, ...]]
    name: Optional[str] = None
    dataset: str = "fineweb_english_1B"
    method: str = "siiSoftPlus"
    divergence: str = "kl"
    order: int = 3
    iters: int = 2000
    rank: int = 150
    shared_factors: Set[Tuple[int, int]] = field(default_factory=lambda: {(1, 2)})
    subsample_frac: float = 0.25

    def _norm_dim(self):
        return tuple(int(x) for x in self.dim.split("-")) if isinstance(self.dim, str) else self.dim

    def _norm_sf(self):
        return tuple(tuple(p) for p in self.shared_factors) if self.shared_factors else None

    def _as_run_config(self) -> RunConfig:
        return RunConfig(
            exp=ExperimentConfig(
              dim=self._norm_dim(), name=self.name, dataset=self.dataset,
              method=self.method, divergence=self.divergence, order=self.order,
              rank=(self.rank,) * self.order,
              subsample_frac=self.subsample_frac,
              shared_factors=self._norm_sf(),
            ),
            train=TrainingConfig(
              n_iter_max=self.iters,
            ),
            eval=EvalConfig(),
        )

    @property
    def stem(self) -> str:
        return self._as_run_config().model_filename().removesuffix(".pt")

    @property
    def log_path(self):
        return self._as_run_config().artifact_paths()["log"]

    @property
    def checkpoint_dir(self):
        return self._as_run_config().artifact_paths()["checkpoint_dir"]

    @property
    def vocab_path(self):
        return (DATA_DIR / "tensors" / self.dataset /
                "vocabularies" / _vocab_filename(self.order, self._norm_dim(), shared_factors=self._norm_sf()))

    def load_tucker(self, map_location="cpu", tier1=False):
        from tensormet.tucker_tensor import TuckerDecomposition
        return TuckerDecomposition.load_from_disk(
            name=self.name,
            dataset=self.dataset,
            dims=self.dim,
            method=self.method,
            divergence=self.divergence,
            order=self.order,
            shared_factors=self.shared_factors,
            subsample_frac=self.subsample_frac,
            iterations=self.iters,
            rank=self.rank,
            map_location=map_location,
            tier1=tier1,
        )

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

    # sentence-boundary padding for n-gram creation:
    # left-pad each sentence with n-1 <s> tokens and right-pad with one </s>.
    pad_sentences: bool = True

    # logging
    log_every_s: float = 30.0

    # optional custom path label (replaces dataset+config in output paths)
    name: Optional[str] = None

@dataclass(frozen=True)
class HFStreamConfig:
    """How to stream texts from a HF dataset."""
    path: str
    config: Optional[str]
    split: str = "train"
    text_column: str = "text"


@dataclass(frozen=True)
class VectorRunConfig:
    exp: VectorExperimentConfig
    hf: HFStreamConfig

    def _path_label(self) -> str:
        """Label used in output paths: custom name if set, otherwise '{dataset}_{config}'."""
        if self.exp.name:
            return self.exp.name
        dataset = self.hf.path.replace("/", "-").strip()
        config = (self.hf.config or "").replace("/", "-").strip()
        return f"{dataset}_{config}"

    def output_dir(self) -> Path:
        label = self._path_label()
        if parse_raw_ngram_orders(self.exp.type) is not None:
            return self.exp.output_dir / f"raw_ngrams_{label}_{self.exp.target_vectors}"
        if parse_ngram_orders(self.exp.type) is not None:
            return self.exp.output_dir / f"ngrams_{label}_{self.exp.target_vectors}"
        return self.exp.output_dir / f"{label}_{self.exp.target_vectors}"

    def ngram_dir(self, n: int, *, raw: bool = False, padded: bool = False) -> Path:
        """Per-order n-gram parquet directory.

        When ``padded`` is set, a ``bos-eos`` tag is inserted so sentence-boundary
        padded vectors live in their own directory and never clobber unpadded ones.
        The tag is a path label only; the actual tokens written are ``<s>``/``</s>``.
        """
        label = self._path_label()
        prefix = f"{n}-gram-raw" if raw else f"{n}-gram"
        if padded:
            prefix += "-bos-eos"
        return self.exp.output_dir / f"{prefix}-{label}_{self.exp.target_vectors}"


@dataclass(frozen=True)
class PopulationExperimentConfig:
    dataset: str = "fineweb-en"
    top_ks: Tuple[int, ...] = (1000, 2000, 4000, 6000)
    top_ks_asymmetric: Optional[Tuple[Tuple[int, ...], ...]] = None
    cols_to_build: Tuple[str, ...] = ("root", "nsubj", "obj")
    shared_factors: Optional[Tuple[Tuple[int, int], ...]] = None
    min_mode_ks: Optional[Tuple[int, ...]] = None  # per-mode minimum vocab floor, indexed by mode
    ensured_vocab: Optional[Tuple[str, ...]] = None  # tokens pinned into the shared vocab by name
    batch_rows: int = 256_000
    batch_readahead: int = 4
    fragment_readahead: int = 2
    max_workers: int = 0        # 0 = auto → scale to cores (cpu_frac) under the mem ceiling
    cpu_frac: float = 0.5       # fraction of cores to target when max_workers == 0 (1.0 = full node)
    max_mem_gb: Optional[float] = None        # explicit RAM ceiling; auto-detected if None
    mem_per_worker_gb: Optional[float] = None  # per-worker RAM estimate override (tune from RSS)
    shards_per_task: int = 1    # shards bundled per worker task; 1 = finest granularity
    vectors_dir_override: Optional[Path] = None  # bypass dataset-derived path
    data_dir: Path = DATA_DIR
    remove_hapax: bool = False
    tensors_to_build: Optional[Tuple[str, ...]] = None  # None = naming.DEFAULT_METHODS

    def vectors_dir(self) -> Path:
        if self.vectors_dir_override is not None:
            return Path(self.vectors_dir_override)
        return self.data_dir / "vectors" / self.dataset

    def output_dir(self) -> Path:
        return self.data_dir / "tensors" / self.dataset

@dataclass(frozen=True)
class PopulationRunConfig:
    exp: PopulationExperimentConfig


