"""
Single source of truth for artifact filename conventions.

Naming contract
---------------
Model file :  {prefix}{div}_{method}_{order}D_{dims}d{sf}_{rank0}r{ss}{mn}_{iters}i.pt
              (experimental CP family: '{order}D' → 'CP{order}D'; Tucker
               filenames are byte-identical to before the CP feature existed)
Vocab file :  {order}D_{dims}d{sf}.pkl          (legacy: {dims}{sf}.pkl)
Populated  :  {method}_{order}D_{dims}d{sf}.pt  (legacy: {method}_{dims}{sf}.pt)

where:
  prefix = "{name}_"  if name is not None, else ""
  sf     = shared_factor_suffix(nontrivial_linked_groups(...))   e.g. "_01" or ""
  ss     = "_{frac_str}ss"  if subsample_frac != 1.0, else ""
  mn     = "_{max_nnz}mn"   if max_nnz, else ""
  dims   = dim_spec_str(dim)                                     e.g. "1000" or "500-1000-500"
  rank0  = rank if int, else rank[0]
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

from tensormet.utils import dim_spec_str, nontrivial_linked_groups, shared_factor_suffix

_Dim   = Union[int, Tuple[int, ...]]
_Rank  = Union[int, Tuple[int, ...]]
_SF    = Optional[Tuple[Tuple[int, int], ...]]


# ---------------------------------------------------------------------------
# Single source of truth for the set of valid tensor-variant method names.
# Used by population.py (as the set of variants it can build) and by both
# TuckerDecomposition.load_from_disk / SparseTupleTensor.load_from_disk (as
# the whitelist for what can be loaded back).
# ---------------------------------------------------------------------------
ALL_METHODS = [
    "counting", "countingLog", "countingLogEps",
    "probLog", "probLogSoftPlus", "probLogShifted",
    "sii", "siiSoftPlus", "siiShifted",
    "sc",  "scSoftPlus",  "scShifted", "scSoftPlusFlat",
]

# Default subset population builds when tensors_to_build is not given. Kept
# deliberately small: at the 1B-sentence scale every extra variant costs
# ~36 bytes/nnz on scratch (the full 13-variant set exceeded the quota).
# Pass --tensors-to-build explicitly to build anything outside this set.
DEFAULT_METHODS = ["countingLog", "countingLogEps", "scSoftPlus"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prefix(name: Optional[str]) -> str:
    return f"{name}_" if name else ""


def _ss(subsample_frac: float) -> str:
    return f"_{str(subsample_frac).replace('.', 'p')}ss" if subsample_frac != 1.0 else ""


def _mn(max_nnz: Optional[int]) -> str:
    return f"_{int(max_nnz)}mn" if max_nnz else ""


def _sf(shared_factors: _SF, order: int) -> str:
    return shared_factor_suffix(nontrivial_linked_groups(shared_factors, num_factors=order))


def _r0(rank: _Rank) -> int:
    """First (or only) rank value; returns 0 for empty tuples."""
    if isinstance(rank, int):
        return rank
    return rank[0] if rank else 0


def _order_tag(order: int, decomposition: str = "tucker", solver: str = "mu") -> str:
    """Order fragment of the model stem: '{order}D' for MU Tucker (unchanged,
    keeps every existing filename byte-identical), 'CP{order}D' for the
    experimental CP family, 'SGD{order}D' for the SGD solver, so
    the artifact families never collide (an SGD run can therefore never scan up
    MU checkpoints on resume, and vice versa). 'SGDCP' is reserved but the
    combination is rejected at fit time."""
    base = f"CP{order}D" if decomposition == "cp" else f"{order}D"
    return f"SGD{base}" if solver == "sgd" else base


# ---------------------------------------------------------------------------
# Model filenames
# ---------------------------------------------------------------------------

def model_stem(
    divergence: str,
    method: str,
    order: int,
    dim: _Dim,
    rank: _Rank,
    n_iter_max: int,
    *,
    name: Optional[str] = None,
    shared_factors: _SF = None,
    subsample_frac: float = 1.0,
    max_nnz: Optional[int] = None,
    decomposition: str = "tucker",
    solver: str = "mu",
) -> str:
    """Return the model filename stem (without .pt extension)."""
    return (
        f"{_prefix(name)}{divergence}_{method}_{_order_tag(order, decomposition, solver)}_"
        f"{dim_spec_str(dim)}d{_sf(shared_factors, order)}_"
        f"{_r0(rank)}r{_ss(subsample_frac)}{_mn(max_nnz)}_{n_iter_max}i"
    )


def model_filename(
    divergence: str,
    method: str,
    order: int,
    dim: _Dim,
    rank: _Rank,
    n_iter_max: int,
    *,
    name: Optional[str] = None,
    shared_factors: _SF = None,
    subsample_frac: float = 1.0,
    max_nnz: Optional[int] = None,
    decomposition: str = "tucker",
    solver: str = "mu",
) -> str:
    """Return the full model filename (e.g. 'fr_siiSoftPlus_3D_1000d_100r_300i.pt')."""
    return model_stem(
        divergence, method, order, dim, rank, n_iter_max,
        name=name, shared_factors=shared_factors, subsample_frac=subsample_frac,
        max_nnz=max_nnz, decomposition=decomposition, solver=solver,
    ) + ".pt"


def candidate_stems(
    divergence: str,
    method: str,
    order: int,
    dim: _Dim,
    rank: _Rank,
    *,
    name: Optional[str] = None,
    shared_factors: _SF = None,
    subsample_frac: float = 1.0,
    max_nnz: Optional[int] = None,
    decomposition: str = "tucker",
    solver: str = "mu",
) -> list[str]:
    """
    Return filename prefixes in descending priority order for directory scans.

    Each prefix ends with "_"; append "*i.pt" (or "*i_config.json") to glob.

    Priority:
      1. new naming with shared-factor suffix   (canonical)
      2. new naming without shared-factor suffix (fallback when sf was added later)
      3. legacy naming: no order prefix          (pre-{order}D era)

    For decomposition="cp" the order fragment becomes 'CP{order}D' and there
    is no legacy fallback (no CP artifacts predate this naming), so the list
    is just [new] (+ [new_no_sf] when a shared-factor suffix applies). The same
    applies to solver="sgd" ('SGD{order}D' fragment, no legacy fallback).
    """
    p   = _prefix(name)
    sf  = _sf(shared_factors, order)
    ss  = _ss(subsample_frac)
    mn  = _mn(max_nnz)
    d   = dim_spec_str(dim)
    r0  = _r0(rank)
    ot  = _order_tag(order, decomposition, solver)

    new        = f"{p}{divergence}_{method}_{ot}_{d}d{sf}_{r0}r{ss}{mn}_"
    new_no_sf  = f"{p}{divergence}_{method}_{ot}_{d}d_{r0}r{ss}{mn}_"
    legacy     = f"{p}{divergence}_{method}_{d}d_{r0}r_"

    # When sf is empty, new == new_no_sf; deduplicate.
    stems = [new]
    if sf:
        stems.append(new_no_sf)
    if decomposition != "cp" and solver != "sgd":
        stems.append(legacy)
    return stems


# ---------------------------------------------------------------------------
# Vocabulary filenames
# ---------------------------------------------------------------------------

def vocab_filename(
    order: int,
    dim: _Dim,
    *,
    shared_factors: _SF = None,
) -> str:
    """Return the vocabulary filename (e.g. '3D_1000d.pkl')."""
    return f"{order}D_{dim_spec_str(dim)}d{_sf(shared_factors, order)}.pkl"


def vocab_filename_legacy(
    dim: _Dim,
    *,
    shared_factors: _SF = None,
    order: int = 3,
) -> str:
    """Return the legacy vocabulary filename (no order prefix, e.g. '1000.pkl')."""
    return f"{dim_spec_str(dim)}{_sf(shared_factors, order)}.pkl"


# ---------------------------------------------------------------------------
# Populated tensor filenames
# ---------------------------------------------------------------------------

def populated_filename(
    method: str,
    order: int,
    dim: _Dim,
    *,
    shared_factors: _SF = None,
) -> str:
    """Return the populated tensor filename (e.g. 'siiSoftPlus_3D_1000d.pt')."""
    return f"{method}_{order}D_{dim_spec_str(dim)}d{_sf(shared_factors, order)}.pt"


def populated_filename_legacy(
    method: str,
    dim: _Dim,
    *,
    shared_factors: _SF = None,
    order: int = 3,
) -> str:
    """Return the legacy populated tensor filename (no order prefix)."""
    return f"{method}_{dim_spec_str(dim)}{_sf(shared_factors, order)}.pt"
