import requests
import os
import time
from pathlib import Path
import torch
import pickle
import multiprocessing
from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager
from threadpoolctl import threadpool_limits
import json
import sys
from typing import Any, Dict, Optional, Union, IO, Tuple
from datetime import datetime, UTC
import logging

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

data_path = os.getenv("SCRATCH_DATA") or os.getenv("DATA")
if data_path is None:
    raise EnvironmentError("Neither SCRATCH_DATA nor DATA environment variable is set")
DATA_DIR = Path(data_path)


def guarded_cupy_import(check_cuda: bool = True) -> Tuple[Optional[object], Optional[object]]:
    """
    Try to import cupy and cupyx.scipy.sparse. If anything fails or no CUDA device is visible,
    return (None, None) and emit a warning via logging.
    """
    logger = logging.getLogger(__name__)
    try:
        import cupy as cp  # type: ignore
    except Exception:
        logger.warning("cupy not installed; falling back to CPU (cp/cpx_sparse disabled).")
        return None, None

    try:
        import cupyx.scipy.sparse as cpx_sparse  # type: ignore
    except Exception:
        cpx_sparse = None

    if check_cuda:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                logger.warning("cupy installed but no GPU visible; falling back to CPU (cp/cpx_sparse disabled).")
                return None, None
        except Exception:
            logger.warning("cupy present but CUDA access failed; falling back to CPU (cp/cpx_sparse disabled).")
            return None, None

    return cp, cpx_sparse

def notify_discord(message, job_finished=True):
    try:

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        prefix = "*Job finished*\n📅" if job_finished else ""
        payload = {
            "content": f"{prefix} {timestamp}\n{message}"
        }
        requests.post(WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print("Could not send Discord notification:", e)


def readonly_dispatch(p: Path, tier1: bool=False) -> Path:
    """
    If tier1 is enabled, make sure 'read-only' paths are under /readonly.
    - Works for absolute paths like /data/... -> /readonly/data/...
    - Leaves relative paths alone (so user can pass local relative files).
    - If path already starts with /readonly, no-op.
    """
    if not tier1:
        return p

    p = Path(p)
    try:
        # normalize without resolving symlinks (no FS access)
        s = p.as_posix()
    except Exception:
        return p

    # only on absolute paths, otherwise our data will get lost
    if not p.is_absolute():
        return p

    if s.startswith("/readonly/") or s == "/readonly":
        return p

    # prepend /readonly to the absolute path
    final_path = Path("/readonly") / p.relative_to("/")
    print("read from", final_path)
    return final_path



def print_elapsed_time(start_time, message=""):
    """Prints the elapsed time since start_time."""
    now = time.time()
    # if the message starts with indents (tabs), add the same number to the elapsed time print
    tabs = ""
    for char in message:
        if char == "\t":
            tabs += "\t"
        else:
            break
    elapsed_time = now - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    if message:
        print(message)
    print(f"{tabs}Elapsed time: {int(minutes)} minutes and {seconds} seconds.")
    return now

def select_gpu(gpu_id=None, n_gpus: int = 1):
    """
    Select GPU(s) for the current process and set CUDA_VISIBLE_DEVICES.

    Modes:
      n_gpus=1, gpu_id=None   — auto-select the single least-used GPU
      n_gpus=1, gpu_id=int    — pin to that specific physical GPU
      n_gpus>1, gpu_id=None   — auto-select the N least-used GPUs
      n_gpus>1, gpu_id=list   — pin to those specific physical GPUs

    After this call, logical device indices 0…n_gpus-1 map to the selected
    physical GPUs via CUDA_VISIBLE_DEVICES remapping.

    Returns torch.device(0) (the primary logical device).
    """
    if torch.cuda.device_count() == 0:
        print("No GPU available.")
        return torch.device("cpu")

    # --- explicit pin ---
    if gpu_id is not None:
        ids = [gpu_id] if isinstance(gpu_id, int) else list(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in ids)
        print(f"Pinned to GPU(s): {ids}")
        return torch.device(0)

    # --- auto-select n_gpus least-used GPUs ---
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        mem_used = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used.append((mem_info.used, i))
        pynvml.nvmlShutdown()

        mem_used.sort()  # ascending by used memory → least loaded first
        n_select = min(n_gpus, len(mem_used))
        selected = [idx for _, idx in mem_used[:n_select]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in selected)
        for rank, (used, phys) in enumerate(mem_used[:n_select]):
            print(f"  GPU rank {rank} → physical GPU {phys}  "
                  f"({used / (1024 ** 2):.0f} MB used)")
        return torch.device(0)
    except Exception as e:
        print("Could not select GPU automatically:", e)
        return torch.device(0)

def torch_or_pickle_load(path, map_location="cpu"):
    """Tries to load a torch-saved file, if fails, tries pickle."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
        # Move tensors to the specified device if not CPU
        device = torch.device(map_location)

    except RuntimeError as e:
        with open(path, "rb") as f:
            return pickle.load(f)

def tree_to_device(x, device):
    """Helps to move (core,factors) objects to the right device."""
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, SparseCOOTensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: tree_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        converted = [tree_to_device(v, device) for v in x]
        return type(x)(converted)
    else:
        # print(f"Warning: could not move object of type {type(x)} to device {device}.")
        raise ValueError("Unsupported type for tree_to_device.")


_INT64_MAX = (1 << 63) - 1


class SparseCOOTensor:
    """
    Lightweight COO sparse tensor that avoids PyTorch's int64 numel overflow
    for very high-order tensors (e.g. 5-gram with top_k=10000, where
    prod(shape) = 10^20 > int64_max).

    Exposes the same interface used in this codebase as torch.sparse_coo_tensor:
    .indices(), .values(), ._nnz(), .coalesce(), .size(), .shape, .is_sparse,
    and .to(device).  Serialises transparently via torch.save / torch.load.
    """
    is_sparse = True

    def __init__(self, indices: torch.Tensor, values: torch.Tensor, size: tuple):
        # indices: (ndim, nnz) long tensor; values: (nnz,) float tensor
        self._indices = indices
        self._values = values
        self._size = tuple(int(d) for d in size)

    # --- core interface ---

    def indices(self) -> torch.Tensor:
        return self._indices

    def values(self) -> torch.Tensor:
        return self._values

    def _nnz(self) -> int:
        return int(self._values.shape[0])

    def coalesce(self) -> "SparseCOOTensor":
        # Entries are built from a Counter (unique keys), so no duplicates exist.
        return self

    def size(self) -> torch.Size:
        return torch.Size(self._size)

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self._size)

    def to(self, device) -> "SparseCOOTensor":
        return SparseCOOTensor(self._indices.to(device), self._values.to(device), self._size)

    def __repr__(self) -> str:
        return (
            f"SparseCOOTensor(shape={self._size}, nnz={self._nnz()}, "
            f"dtype={self._values.dtype})"
        )

def compute_num_threads(max_cpu_frac: float = 0.75, min_threads: int = 1) -> int:
    """Deterministic thread budget based on available CPUs."""
    try:
        n_cores = multiprocessing.cpu_count()
    except NotImplementedError:
        n_cores = os.cpu_count() or 1
    return max(min_threads, int(n_cores * max_cpu_frac))

@dataclass(frozen=True)
class ThreadBudget:
    n_threads: int

    @contextmanager
    def limit(self):
        """Context manager to apply the thread limit."""
        with threadpool_limits(self.n_threads):
            yield



def einsum_letters(n):
    # enough for typical tensor orders; extend if you ever go beyond 52
    letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if n > len(letters):
        raise ValueError(f"Tensor order {n} too large for einsum-letter helper ({len(letters)} available).")
    return letters[:n]

def voc_index(role: str) -> str:
    return f"{role}2i"

def extract_roles_from_vocab(vocab):
    roles = [k[len("vocab_"):] for k in vocab.keys() if k.startswith("vocab_")]
    if not roles:
        return ["verb", "subject", "object"] #safe default
    else:
        return roles

# --- Logging utilities ---
def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")
def utc_now_iso() -> str:
    """Returns the current UTC time in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


class _TeeStream:
    """
    A file-like stream that duplicates writes to multiple streams.
    Works well for capturing print() + tqdm (stderr) into a log file.
    """
    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        # Be tolerant: some libraries write None or empty strings
        if not data:
            return 0
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                # Don't crash the training loop because the logger stream failed
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        # Helps tqdm behave as if it still has a TTY when running in a terminal
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)


@contextmanager
def tee_output(
    log_file: Optional[Union[str, Path]],
    *,
    mode: str = "a",
    encoding: str = "utf-8",
):
    """
    Tee *all* stdout and stderr output to `log_file` (and still show it in the console).

    Usage:
        with tee_output("run.log"):
            print("captured")
            tqdm(...)

    If log_file is None, this is a no-op.
    """
    if log_file is None:
        yield
        return

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    f = open(log_path, mode=mode, encoding=encoding, buffering=1)  # line-buffered
    old_out, old_err = sys.stdout, sys.stderr

    try:
        sys.stdout = _TeeStream(old_out, f)
        sys.stderr = _TeeStream(old_err, f)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = old_out, old_err
        try:
            f.close()
        except Exception:
            pass

# factor sharing
def linked_factor_groups(num_factors: int, shared_factors=None) -> list[list[int]]:
    """
    Convert pairwise links like {(1,2), (2,3)} into connected groups:
    [[1,2,3], [0], ...]
    """
    parent = list(range(num_factors))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if shared_factors:
        for a, b in shared_factors:
            if not (0 <= a < num_factors and 0 <= b < num_factors):
                raise ValueError(
                    f"Invalid shared_factors entry {(a, b)} for {num_factors} factors."
                )
            union(a, b)

    groups = defaultdict(list)
    for i in range(num_factors):
        groups[find(i)].append(i)

    return list(groups.values())

def nontrivial_linked_groups(shared_factors, num_factors: int = 3) -> list[list[int]]:
    """
    Normalize pairwise shared_factors into connected groups, matching population code.
    """
    if not shared_factors:
        return []

    groups = linked_factor_groups(num_factors=num_factors, shared_factors=shared_factors)
    return [group for group in groups if len(group) > 1]

def shared_factor_suffix(shared_factors) -> str:
    if not shared_factors:
        return ""
    parts = ["shared" + "".join(map(str, pair)) for pair in sorted(shared_factors)]
    return "_" + "_".join(parts)