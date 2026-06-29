"""
Node-local staging helpers for HPC (``--hpc``) runs.

On a shared parallel filesystem (e.g. GPFS) many concurrent array tasks writing
into the same decomposition directory cause metadata-lock contention that stalls
iterations for minutes at a time. Under ``--hpc`` every per-run artifact write
goes to node-local ``$TMPDIR`` instead; this module owns the two movements back
to durable storage:

  * :func:`mirror_checkpoint` — during the run, copy each periodic checkpoint
    (and its errors/fitness siblings) to GPFS so a walltime kill — which never
    runs the end-of-job copy-back — loses at most one checkpoint interval and
    stays resumable.
  * :func:`stage_artifacts_back` — once at job end (clean exit or caught
    exception), flush the full staged run tree to its canonical GPFS location.

Both build on :func:`copy_artifact`, an atomic single-file copy so a concurrent
resumer never observes a half-written file.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional

# Keys in an artifact_paths() dict that are single files (not the checkpoint dir).
_FILE_KEYS = ("model", "errors", "fitness", "fitness_json", "timing_json", "config", "log")


def copy_artifact(src, dst) -> bool:
    """Best-effort atomic copy of one staged artifact to its durable location.

    The copy is staged into a sibling ``*.tmp.<pid>`` then ``os.replace``\\d into
    place (atomic on the same filesystem), so a concurrent resumer reading the
    same directory never observes a half-written file. Returns True on success,
    False if the source is missing, identical to the destination, or the copy
    failed (failures are logged, never raised — callers run inside ``finally``
    blocks and must not have the original error masked).
    """
    src, dst = Path(src), Path(dst)
    tmp = None
    try:
        if src == dst or not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)  # atomic same-fs rename
        return True
    except Exception as e:
        print(f"[hpc] WARNING: mirror {src} -> {dst} failed: {e}")
        if tmp is not None:
            try:
                Path(tmp).unlink()
            except OSError:
                pass
        return False


def mirror_checkpoint(
    paths: Dict[str, Path],
    mirror_paths: Optional[Dict[str, Path]],
    ckpt_name: str,
) -> None:
    """Mirror a just-written checkpoint and its errors/fitness siblings to GPFS.

    ``paths`` are the staged ($TMPDIR) artifact paths, ``mirror_paths`` the
    canonical (GPFS) ones. No-op when ``mirror_paths`` is None (i.e. not HPC mode).
    """
    if mirror_paths is None:
        return
    copy_artifact(paths["checkpoint_dir"] / ckpt_name,
                  mirror_paths["checkpoint_dir"] / ckpt_name)
    for key in ("errors", "fitness", "fitness_json"):
        copy_artifact(paths[key], mirror_paths[key])


def stage_artifacts_back(
    work_paths: Dict[str, Path],
    final_paths: Dict[str, Path],
) -> None:
    """Copy staged ($TMPDIR) run artifacts back to their canonical GPFS paths.

    Best-effort and per-file so that, when invoked from a ``finally`` block, a
    copy failure never masks the original training error. Entries where staged
    and canonical coincide (staging is a no-op) are skipped.
    """
    for key in _FILE_KEYS:
        src = work_paths.get(key)
        dst = final_paths.get(key)
        if src is None or dst is None:
            continue
        copy_artifact(src, dst)

    # Checkpoint directory: copy the whole tree so resume can pick it up later.
    src_dir = work_paths.get("checkpoint_dir")
    dst_dir = final_paths.get("checkpoint_dir")
    if src_dir is not None and dst_dir is not None and src_dir != dst_dir and Path(src_dir).exists():
        try:
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        except Exception as e:
            print(f"[hpc] WARNING: failed to copy checkpoints {src_dir} -> {dst_dir}: {e}")

    print(f"[hpc] copied staged artifacts back to {final_paths['model'].parent}")
