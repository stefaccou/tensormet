import requests
import socket
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
from typing import Any, Dict
from datetime import datetime, UTC


WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

def notify_discord(message, job_finished=True):
    try:

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "content": f"{'*Job finished * \n📅'*job_finished} {timestamp}\n{message}"
        }
        requests.post(WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print("Could not send Discord notification:", e)


def find_project_root(start: Path | None = None) -> Path:
    """
    Walk up from 'start' until we find a marker that defines the project root.
    Markers can be: .git, pyproject.toml, or a custom '.project-root' file.
    """
    if start is None:
        start = Path.cwd().resolve()
    start = start.resolve()

    for parent in [start, *start.parents]:
        if any((parent / marker).exists() for marker in (".venv", ".git", "pyproject.toml", ".project-root")):
            return parent

    raise RuntimeError("Could not find project root – add a marker like '.project-root' at the root.")

# convenience globals
PROJECT_ROOT = find_project_root(Path(__file__).resolve())
PROJECT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = Path("/home/local/stefa/data")

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