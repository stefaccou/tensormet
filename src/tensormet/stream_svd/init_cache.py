"""
On-disk cache for the svd_stream init, in ``tensors/{dataset}/decomposition/init_cache/``.

One file per group of tied modes holds the raw basis (U, s) at the largest rank
computed so far; any rank k <= K is served by truncation, which is exact
because NNDSVD treats every column on its own. A Tucker core gets its own file
and is served by slicing its leading sub-block, for the same reason.

The key holds only what changes the result: the populated tensor, the group,
the sketch settings (power iterations q, oversampling o: "q6o100") and the
random state (it seeds the sketch). Rank, decomposition, n_iter, subsampling
and n_gpus are not part of it. Newer files also hold the per-column residuals.
The core is built from the factors after ``--svd-fill``, so a non-zero fill
has its own core files ("core_fill1_r..."); the basis files are shared.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

CACHE_DIRNAME = "init_cache"


def _group_tag(group: Sequence[int]) -> str:
    return "m" + ".".join(str(m) for m in group)


def fingerprint(nnz: int, sum_x: float, shape) -> dict:
    return {"nnz": int(nnz), "sum_x": float(sum_x), "shape": [int(s) for s in shape]}


def _same_tensor(a: dict, b: dict) -> bool:
    return (a["nnz"] == b["nnz"] and a["shape"] == b["shape"]
            and abs(a["sum_x"] - b["sum_x"]) <= 1e-6 * max(1.0, abs(b["sum_x"])))


def _atomic_savez(path: Path, **arrays) -> None:
    # Temp file + rename: concurrent jobs never read a half-written file.
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _read_meta(npz) -> dict:
    return json.loads(str(npz["meta"]))


class InitCache:
    """Basis and core files for one populated tensor and one sketch setting."""

    def __init__(self, directory, tensor_stem: str, sketch_tag: str, fill: float = 0.0):
        self.directory = Path(directory)
        self.prefix = f"svdstream_{tensor_stem}"
        self.sketch_tag = sketch_tag
        self.core_tag = f"core_fill{float(fill):g}" if fill else "core"

    @classmethod
    def from_config(cls, cfg, sketch_tag: str, fill: float = 0.0) -> "InitCache":
        from tensormet.naming import populated_filename
        stem = populated_filename(cfg.exp.method, cfg.exp.order, cfg.exp.dim,
                                  shared_factors=cfg.exp.shared_factors)
        stem = stem[:-3] if stem.endswith(".pt") else stem
        # Canonical (GPFS) location, also under HPC staging, so later jobs find it.
        return cls(Path(cfg.output_dir(staged=False)) / CACHE_DIRNAME, stem, sketch_tag, fill)

    # --- basis -----------------------------------------------------------
    def basis_path(self, group: Sequence[int]) -> Path:
        return self.directory / f"{self.prefix}_{_group_tag(group)}_{self.sketch_tag}.npz"

    def load_basis(self, group, k: int, fp: dict):
        """(U[:, :k], s[:k], token, r[:k]) as host arrays, or None on a miss.
        ``r`` (per-column residuals) is None for files from before it was stored."""
        path = self.basis_path(group)
        if not path.exists():
            return None
        try:
            with np.load(path) as z:
                meta = _read_meta(z)
                if not _same_tensor(meta["fingerprint"], fp):
                    print(f"[svd_stream] cache {path.name}: tensor changed, recomputing")
                    return None
                if meta["K"] < k:
                    print(f"[svd_stream] cache {path.name}: holds K={meta['K']} < {k}, recomputing")
                    return None
                U, s = z["U"][:, :k], z["s"][:k]
                r = np.asarray(z["r"][:k]) if "r" in z.files else None
        except Exception as exc:  # unreadable file: recompute, don't fail the run
            print(f"[svd_stream] cache {path.name} unreadable ({exc!r}), recomputing")
            return None
        print(f"[svd_stream] cache hit {path} (K={meta['K']}, k={k}, "
              f"saved {meta.get('seconds', 0):.0f}s)")
        return U, s, meta["token"], r

    def timings(self, group) -> List[dict]:
        """meta of every cached basis of this group (any sketch setting, any seed), newest first."""
        out = []
        for path in self.directory.glob(f"{self.prefix}_{_group_tag(group)}_q*.npz"):
            if "_core_" in path.name:
                continue
            try:
                with np.load(path) as z:
                    out.append(_read_meta(z))
            except Exception:
                continue
        return sorted(out, key=lambda m: m.get("created", ""), reverse=True)

    def save_basis(self, group, U, s, fp: dict, seconds: float, residuals=None) -> str:
        token = uuid.uuid4().hex
        meta = {"K": int(U.shape[1]), "group": list(map(int, group)), "fingerprint": fp,
                "sketch": self.sketch_tag, "seconds": float(seconds), "token": token,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        path = self.basis_path(group)
        extra = {} if residuals is None else {"r": np.asarray(residuals)}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            _atomic_savez(path, U=np.asarray(U), s=np.asarray(s), meta=np.array(json.dumps(meta)), **extra)
            print(f"[svd_stream] saved {path} (K={meta['K']})")
        except OSError as exc:
            print(f"[svd_stream] could not save {path} ({exc!r}); continuing without cache")
        return token

    # --- core ------------------------------------------------------------
    def _core_glob(self, groups) -> str:
        tag = "_".join(_group_tag(g) for g in groups)
        return f"{self.prefix}_{tag}_{self.sketch_tag}_{self.core_tag}_r*.npz"

    def core_path(self, groups, ranks) -> Path:
        tag = "_".join(_group_tag(g) for g in groups)
        r = "-".join(str(int(x)) for x in ranks)
        return self.directory / f"{self.prefix}_{tag}_{self.sketch_tag}_{self.core_tag}_r{r}.npz"

    def load_core(self, groups, ranks, tokens: List[str], fp: dict) -> Optional[np.ndarray]:
        """Leading sub-block of the smallest stored core that covers ``ranks``
        and was built from the current bases (same tokens)."""
        if not self.directory.exists():
            return None
        best = None
        for path in self.directory.glob(self._core_glob(groups)):
            try:
                with np.load(path) as z:
                    meta = _read_meta(z)
                    if (meta["tokens"] != list(tokens) or not _same_tensor(meta["fingerprint"], fp)
                            or any(R < r for R, r in zip(meta["ranks"], ranks))):
                        continue
                    size = int(np.prod(meta["ranks"]))
                    if best is None or size < best[0]:
                        best = (size, path)
            except Exception:
                continue
        if best is None:
            return None
        with np.load(best[1]) as z:
            core = z["core"][tuple(slice(0, int(r)) for r in ranks)]
        print(f"[svd_stream] core cache hit {best[1].name}")
        return np.ascontiguousarray(core)

    def save_core(self, groups, core, ranks, tokens: List[str], fp: dict, seconds: float) -> None:
        meta = {"ranks": [int(r) for r in ranks], "tokens": list(tokens), "fingerprint": fp,
                "seconds": float(seconds), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        path = self.core_path(groups, ranks)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            _atomic_savez(path, core=np.asarray(core), meta=np.array(json.dumps(meta)))
            print(f"[svd_stream] saved {path}")
        except OSError as exc:
            print(f"[svd_stream] could not save {path} ({exc!r}); continuing without cache")
