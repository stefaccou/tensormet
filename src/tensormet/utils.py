import requests
import os
import time
from pathlib import Path
import torch
import pickle
import multiprocessing
from dataclasses import dataclass
from contextlib import contextmanager
from threadpoolctl import threadpool_limits
import json
import sys
from typing import Any, Dict, Optional, Union, IO
from datetime import datetime, UTC

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

data_path = os.getenv("SCRATCH_DATA") or os.getenv("DATA")
if data_path is None:
    raise EnvironmentError("Neither SCRATCH_DATA nor DATA environment variable is set")
DATA_DIR = Path(data_path)


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

def select_gpu(gpu_id=None):
    """Selects the least used GPU or a specific one if gpu_id is provided."""
    if torch.cuda.device_count() == 0:
        print("No GPU available.")
        return torch.device("cpu")

    elif gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device(gpu_id)
        return device
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        min_used_mem = float('inf')
        best_gpu = 0

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mem = mem_info.used

            if used_mem < min_used_mem:
                min_used_mem = used_mem
                best_gpu = i

        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        print(f"Selected GPU {best_gpu} with {min_used_mem / (1024 ** 2):.2f} MB used memory.")
        pynvml.nvmlShutdown()
        device = torch.device(best_gpu)
        return device
    except Exception as e:
        print("Could not select GPU automatically:", e)

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
    elif isinstance(x, dict):
        return {k: tree_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        converted = [tree_to_device(v, device) for v in x]
        return type(x)(converted)
    else:
        # print(f"Warning: could not move object of type {type(x)} to device {device}.")
        raise ValueError("Unsupported type for tree_to_device.")

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