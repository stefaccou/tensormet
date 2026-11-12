"""
families.py — what differs between the decomposition families, in one table.

The training loop (training.py) and the solvers (solvers.py) never branch on
``tucker`` / ``tt`` / ``utt`` / ``cp``: everything that differs is a field of
``Family``, built here from functions that live next to each family's kernels.
Binary Tucker (EXPERIMENTAL) is the Tucker entry with a different init and
sharded routing, see ``get_family``.

Imports of the families' own modules are lazy, so a Tucker run never touches
CP / TT / utt code.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

from tensormet.routing import get_update_routing_step, get_sharded_update_routing_step
from tensormet.utils import make_lazy_cupy_pair

cp, _cpx_sparse = make_lazy_cupy_pair()


@dataclass(frozen=True)
class Family:
    # (tensor, shape, rank, modes, cfg, thread_budget) -> (core, factors); CuPy.
    initialize: Callable
    # (cfg, log_step, core_power) -> UpdateRouting
    routing: Callable
    # (sst, cfg, log_step, core_power) -> UpdateRouting, for n_gpus > 1
    sharded_routing: Callable
    # (core, factors, vocab, roles) -> the decomposition the semantic eval scores
    make_decomposition: Callable
    # (core, factors) -> the object that gets saved and returned
    container: Callable
    # (core, factors) -> NNZ batch size of the streaming kernels; None = they size their own
    batch_nnz: Optional[Callable]
    # the core is a list (tt: the TT cores, utt: [C, alpha, omega])
    list_core: bool
    # (core, factors, cfg, rank, dtype) -> (core, factors): validate a resumed checkpoint
    check_resume: Callable


# ---------------------------------------------------------------------------
# Tucker
# ---------------------------------------------------------------------------

def _tucker_initialize(tensor, shape, rank, modes, cfg, thread_budget):
    from tensormet.sparse_ops import initialize_nonnegative_tucker
    return initialize_nonnegative_tucker(tensor, shape, rank, modes, cfg.exp.init,
                                         cfg.exp.random_state, thread_budget=thread_budget)


def _tucker_container(core, factors):
    from tensorly.tucker_tensor import TuckerTensor
    return TuckerTensor((core, factors))


def _tucker_decomposition(core, factors, vocab, roles):
    from tensormet.tucker_tensor import TuckerDecomposition
    return TuckerDecomposition(core=core, factors=factors, vocab=vocab, roles=roles)


def _single_routing(decomposition):
    """The single-GPU routing of one family, built from the config."""
    def routing(cfg, log_step, core_power=None):
        e = cfg.exp
        return get_update_routing_step(
            divergence=e.divergence, dim=e.dim, log_step=log_step,
            largedim=cfg.train.largedim, masked=e.objective == "masked",
            decomposition=decomposition, cp_inner_iters=e.cp_inner_iters,
            cp_scooch_kappa=e.cp_scooch_kappa, utt_core_power=core_power, binary=e.binary)
    return routing


def _tucker_sharded_routing(sst, cfg, log_step, core_power=None):
    e = cfg.exp
    return get_sharded_update_routing_step(
        sst, e.divergence, e.dim, log_step, largedim=cfg.train.largedim,
        masked=e.objective == "masked")


def _no_resume_check(core, factors, cfg, rank, dtype):
    return core, factors


# ---------------------------------------------------------------------------
# Tucker-TT hybrid
# ---------------------------------------------------------------------------

def _tt_initialize(tensor, shape, rank, modes, cfg, thread_budget):
    from tensormet.tt_hybrid.tt_ops import initialize_tucker_tt
    e = cfg.exp
    return initialize_tucker_tt(tensor, shape, rank, modes, e.init, e.random_state,
                                tt_rank=e.tt_rank, thread_budget=thread_budget,
                                epsilon=e.epsilon, core_init=e.tt_core_init)


def _tt_sharded_routing(sst, cfg, log_step, core_power=None):
    from tensormet.tt_hybrid.tt_routing import get_sharded_tt_update_routing_step
    return get_sharded_tt_update_routing_step(sst, divergence=cfg.exp.divergence,
                                              log_step=log_step)


def _tt_decomposition(core, factors, vocab, roles):
    from tensormet.tt_hybrid.tt_decomposition import TuckerTTDecomposition
    return TuckerTTDecomposition(tt_cores=core, factors=factors, vocab=vocab, roles=roles)


def _tt_batch_nnz(core, factors):
    from tensormet.tt_hybrid.tt_ops import estimate_batch_nnz_tt
    return estimate_batch_nnz_tt(core, factors)


def _tt_check_resume(core, factors, cfg, rank, dtype):
    from tensormet.tt_hybrid.tt_chain import core_shapes
    expected = core_shapes(rank, cfg.exp.tt_rank)
    if [tuple(int(d) for d in C.shape) for C in core] != expected:
        raise ValueError(
            f"decomposition='tt' resumed a checkpoint whose TT cores have "
            f"shapes {[tuple(C.shape) for C in core]}; expected {expected} "
            f"for rank={rank}, tt_rank={cfg.exp.tt_rank}. Delete it or match the config."
        )
    return core, factors


def _list_container(core, factors):
    # A plain tuple, so no module path ends up inside the pickle and a saved
    # model survives the code moving.
    return (core, factors)


# ---------------------------------------------------------------------------
# Shared-core TT (utt)
# ---------------------------------------------------------------------------

def _utt_initialize(tensor, shape, rank, modes, cfg, thread_budget):
    from tensormet.uniform_tt.utt_ops import initialize_uniform_tt
    e = cfg.exp
    return initialize_uniform_tt(tensor, shape, rank, modes, e.init, e.random_state,
                                 tt_rank=e.tt_rank, thread_budget=thread_budget,
                                 epsilon=e.epsilon, core_init=e.tt_core_init)


def _utt_sharded_routing(sst, cfg, log_step, core_power=None):
    from tensormet.uniform_tt.utt_routing import get_sharded_utt_update_routing_step
    return get_sharded_utt_update_routing_step(sst, divergence=cfg.exp.divergence,
                                               log_step=log_step, core_power=core_power)


def _utt_decomposition(core, factors, vocab, roles):
    from tensormet.uniform_tt.utt_decomposition import UniformTTDecomposition
    return UniformTTDecomposition(tt_cores=core, factors=factors, vocab=vocab, roles=roles)


def _utt_batch_nnz(core, factors):
    from tensormet.uniform_tt.utt_ops import estimate_batch_nnz_utt
    return estimate_batch_nnz_utt(core, factors)


def _utt_check_resume(core, factors, cfg, rank, dtype):
    rho, r = int(cfg.exp.tt_rank), int(rank[0])
    expected = [(rho, r, rho), (rho,), (rho,)]
    if [tuple(int(d) for d in C.shape) for C in core] != expected:
        raise ValueError(
            f"decomposition='utt' resumed a checkpoint whose core has shapes "
            f"{[tuple(C.shape) for C in core]}; expected {expected} "
            f"for rank={rank}, tt_rank={cfg.exp.tt_rank}. Delete it or match the config."
        )
    # Train in the data's dtype: a float64 checkpoint would run ~4x slower.
    if any(x.dtype != dtype for x in core + factors):
        print(f"utt resume: casting the checkpoint from "
              f"{sorted({str(x.dtype) for x in core + factors})} to {dtype}.")
        core = [C.astype(dtype, copy=False) for C in core]
        factors = [f.astype(dtype, copy=False) for f in factors]
    return core, factors


# ---------------------------------------------------------------------------
# CP (EXPERIMENTAL)
# ---------------------------------------------------------------------------

def _cp_initialize(tensor, shape, rank, modes, cfg, thread_budget):
    from tensormet.experimental.CP.cp_ops import initialize_nonnegative_cp
    e = cfg.exp
    # `core` is the λ weight vector.
    return initialize_nonnegative_cp(tensor, shape, rank, modes, e.init, e.random_state,
                                     thread_budget=thread_budget, divergence=e.divergence,
                                     epsilon=e.epsilon)


def _cp_sharded_routing(sst, cfg, log_step, core_power=None):
    from tensormet.experimental.CP.cp_routing import get_sharded_cp_update_routing_step
    e = cfg.exp
    return get_sharded_cp_update_routing_step(
        sst, divergence=e.divergence, log_step=log_step,
        inner_iters=e.cp_inner_iters, scooch_kappa=e.cp_scooch_kappa)


def _cp_container(core, factors):
    from tensorly.cp_tensor import CPTensor
    return CPTensor((core, factors))


def _cp_decomposition(core, factors, vocab, roles):
    from tensormet.experimental.CP.cp_decomposition import CPDecomposition
    return CPDecomposition(weights=core, factors=factors, vocab=vocab, roles=roles)


def _cp_check_resume(core, factors, cfg, rank, dtype):
    if core.ndim != 1:
        # A CP-named checkpoint whose "λ" is not a vector is a Tucker payload;
        # resuming would treat its core as weights.
        raise ValueError(
            f"decomposition='cp' resumed a checkpoint whose core has shape "
            f"{tuple(core.shape)}; expected a 1-D weight vector. This is a "
            f"Tucker payload under a CP filename — delete it and restart."
        )
    return core, factors


# ---------------------------------------------------------------------------
# Binary Tucker (EXPERIMENTAL): the Tucker entry with its own init and routing
# ---------------------------------------------------------------------------

def _binary_initialize(tensor, shape, rank, modes, cfg, thread_budget):
    from tensormet.experimental.binary import prepare_binary
    core, factors = _tucker_initialize(tensor, shape, rank, modes, cfg, thread_budget)
    return prepare_binary(cfg, core, factors, tensor, shape)


def _binary_sharded_routing(sst, cfg, log_step, core_power=None):
    from tensormet.experimental.binary.binary_routing import (
        get_sharded_binary_update_routing_step,
    )
    return get_sharded_binary_update_routing_step(
        sst, divergence=cfg.exp.divergence, log_step=log_step, binary=cfg.exp.binary)


FAMILIES = {
    "tucker": Family(
        initialize=_tucker_initialize, routing=_single_routing("tucker"),
        sharded_routing=_tucker_sharded_routing, make_decomposition=_tucker_decomposition,
        container=_tucker_container, batch_nnz=None, list_core=False,
        check_resume=_no_resume_check),
    "tt": Family(
        initialize=_tt_initialize, routing=_single_routing("tt"),
        sharded_routing=_tt_sharded_routing, make_decomposition=_tt_decomposition,
        container=_list_container, batch_nnz=_tt_batch_nnz, list_core=True,
        check_resume=_tt_check_resume),
    "utt": Family(
        initialize=_utt_initialize, routing=_single_routing("utt"),
        sharded_routing=_utt_sharded_routing, make_decomposition=_utt_decomposition,
        container=_list_container, batch_nnz=_utt_batch_nnz, list_core=True,
        check_resume=_utt_check_resume),
    "cp": Family(
        initialize=_cp_initialize, routing=_single_routing("cp"),
        sharded_routing=_cp_sharded_routing, make_decomposition=_cp_decomposition,
        container=_cp_container, batch_nnz=None, list_core=False,
        check_resume=_cp_check_resume),
}


def get_family(cfg) -> Family:
    """The ``Family`` of ``cfg.exp.decomposition`` (binary Tucker wraps Tucker's)."""
    family = FAMILIES[cfg.exp.decomposition]
    if cfg.exp.binary:
        family = replace(family, initialize=_binary_initialize,
                         sharded_routing=_binary_sharded_routing)
    return family
