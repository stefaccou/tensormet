"""
utt_ops.py — KL multiplicative updates for the shared-core ("uniform") Tucker-TT.

Model, with every factor tied:

    x̂(i_0 … i_{N-1}) = αᵀ S(i_0) ⋯ S(i_{N-1}) ω,     S(i) = Σ_r A[i, r] · C[:, r, :]

The core in the training loop is the plain list ``[C, alpha, omega]``
(three distinct arrays). Blocks and their MU steps:

    A        degree N in A   pooled tied update, undamped (as in TT)
    C        degree N in C   pooled over the N sites, undamped (exponent γ = 1)
    (α, ω)   degree 2        joint step, undamped, then the exact overall
                             scale Σx / Σ_all x̂ (restores the Poisson identity)

All steps are undamped by default, like TT's tied update: right fixed point, no
monotonicity guarantee. For a block that x̂ is homogeneous of degree d in,
exponent 1/d is the MM-safe step (weighted AM-GM bound on x̂ + Jensen on the
log term); ``power`` (C, A) and ``boundary_power`` (α, ω) restore it — 1/N and
1/2. The damped core step barely moved the parameters at production scale.
``AdaptiveCorePower`` (``--utt-core-power auto``) picks the core exponent in
[1/N, 1] per iteration by backtracking (see its docstring).

Same machinery as the per-position TT: every NNZ pass expands [C, α, ω] to an
ordinary TT core list (``expand_uniform``) and runs the tt_hybrid batch
operations on it — ``sites`` / ``left_envs`` / ``right_envs`` / ``site_grad``,
``_scatter_add``, the site GEMM, ``_run_nnz_batches`` and the TT batch
estimator. Per entry a factor pass therefore costs exactly what TT's pooled
factor pass costs; the core needs ONE pass (TT: N), plus one cheap boundary pass.

Gauge: after every A or C step, C is divided by the Perron root λ of
T = Σ_r s[r] M_r and α is multiplied by λᴺ (exact), then α and ω are balanced
to equal ℓ1 mass (exact). The chain then neither blows up nor vanishes at any
order.
"""
from __future__ import annotations

import math
import time

import numpy as np
import tensorly as tl

from tensormet.distance import coo_to_coords
from tensormet.utils import make_lazy_cupy_pair
from tensormet.tt_hybrid.tt_chain import left_envs, right_envs, site_grad, sites
from tensormet.tt_hybrid.tt_ops import (
    MU_RATIO_MAX, _run_nnz_batches, _scatter_add, estimate_batch_nnz_tt,
)
from tensormet.uniform_tt.utt_chain import (
    boundary_sweeps, chain_total, expand_uniform, gauge_scales, transfer_matrix,
)

cp, cpx_sparse = make_lazy_cupy_pair()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tied_factor(factors):
    """The one factor matrix every mode uses. The training loop ties all modes
    to the same object; anything else is not a uniform-TT model."""
    A = factors[0]
    if any(f is not A for f in factors[1:]):
        raise ValueError(
            "decomposition='utt' needs every mode to share one factor matrix "
            "(shared_factors covering all modes)."
        )
    return A


def _check_core(core, n_modes):
    if len(core) != 3 or core[0].ndim != 3 or core[1].ndim != 1 or core[2].ndim != 1:
        raise ValueError("decomposition='utt' expects core = [C (ρ,R,ρ), alpha (ρ,), omega (ρ,)]")
    if n_modes < 2:
        raise ValueError("decomposition='utt' needs order >= 2")


def _colsums(A, epsilon):
    return cp.clip(cp.sum(A, axis=0), a_min=epsilon, a_max=None)


def _closed_forms(core, A, n_modes, epsilon):
    """fp64 column-sum chain: (C64, s, Ls, Rs, P). See utt_chain.boundary_sweeps."""
    C, alpha, omega = (x.astype(cp.float64) for x in core)
    s = _colsums(A, epsilon).astype(cp.float64)
    Ls, Rs = boundary_sweeps(transfer_matrix(C, s), alpha, omega, n_modes)
    P = sum(Ls[k][:, None] * Rs[k + 1][None, :] for k in range(n_modes))
    return C, s, Ls, Rs, P


def utt_sum_all_entries(core, factors, epsilon=1e-12, n_modes=None):
    """Σ over ALL entries of X̂ = αᵀ Tᴺ ω, in fp64."""
    A = factors[0]
    n = len(factors) if n_modes is None else n_modes
    C, alpha, omega = (x.astype(cp.float64) for x in core)
    T = transfer_matrix(C, _colsums(A, epsilon).astype(cp.float64))
    total = chain_total(T, alpha, omega, n)
    return cp.clip(total.astype(A.dtype), a_min=epsilon, a_max=None)


def _xhat_floor(core, A, shape, epsilon):
    """Zero-lock floor for x̂, relative to the mean entry (as in tt_ops)."""
    n_cells = int(np.prod(shape, dtype=object))
    mean = float(utt_sum_all_entries(core, [A], epsilon, n_modes=len(shape))) / n_cells
    return max(float(np.finfo(A.dtype).tiny), epsilon * mean)


def normalize_gauge(core, A, n_modes, perron=True):
    """Exact rescaling of [C, α, ω] in place: C /= λ (Perron root of T), α *= λᴺ,
    then α and ω balanced to equal ℓ1 mass (``utt_chain.gauge_scales``). x̂ is
    unchanged."""
    C, alpha, omega = core
    T = (cp.asnumpy(transfer_matrix(C.astype(cp.float64), cp.sum(A, axis=0).astype(cp.float64)))
         if perron else None)
    lam, alpha_mult, omega_mult = gauge_scales(T, cp.sum(alpha), cp.sum(omega), n_modes)
    if lam is not None:
        core[0] = C / C.dtype.type(lam)
    core[1] = alpha * alpha.dtype.type(alpha_mult)
    core[2] = omega * omega.dtype.type(omega_mult)
    return core


def estimate_batch_nnz_utt(core, factors, **kwargs):
    """TT's batch estimator on the expanded chain: the passes below allocate
    exactly what TT's kernels allocate per entry."""
    return estimate_batch_nnz_tt(expand_uniform(core, len(factors)), factors, **kwargs)


# ---------------------------------------------------------------------------
# NNZ passes (shared by the single-GPU kernels and the shard workers)
# ---------------------------------------------------------------------------

def _batch_chain(tt, A, idxs, start, end, vals, xf):
    """Gathered rows, TT environments, w = x / x̂ and the floored x̂ for one
    batch, exactly as the tt_hybrid kernels build them (x̂ = L[N][:, 0])."""
    n = len(tt)
    mats = [A[idxs[k][start:end]] for k in range(n)]
    S = sites(tt, mats, cp)
    L, R = left_envs(S, cp), right_envs(S, cp)
    xhat = cp.clip(L[n][:, 0], a_min=xf, a_max=None)
    w = cp.minimum(vals[start:end] / xhat, MU_RATIO_MAX)
    return mats, L, R, w, xhat


def _xlogxhat(vals_b, xhat):
    """Σ x log x̂ over one batch, accumulated in fp64 (the adaptive core step
    compares two of these)."""
    return cp.sum(vals_b * cp.log(xhat), dtype=cp.float64)


def factor_pass(idxs, vals, A, core, xf, batch_nnz, verbose=False, desc=None):
    """Pooled factor numerator (V, R): Σ_k Σ_{p : i_k(p)=i} w_p · ∂x̂_p/∂A[i, :].
    The body of tt_kl_tied_factor_update, on the expanded chain.

    Every pass here adds to its result only on the batch body's last lines, so a
    batch that runs out of memory has added nothing and ``_run_nnz_batches``
    can redo it at a smaller size."""
    n = len(idxs)
    tt = expand_uniform(core, n)
    num = cp.zeros_like(A)

    def _body(start, end):
        _mats, L, R, w, _xhat = _batch_chain(tt, A, idxs, start, end, vals, xf)
        part = cp.zeros_like(A)
        for k in range(n):
            Z = site_grad(L[k], tt[k], R[k + 1], cp)
            _scatter_add(part, idxs[k][start:end], w, Z)
        num[...] += part

    if int(vals.size):
        _run_nnz_batches(int(vals.size), batch_nnz, _body, verbose, desc)
    return num


def core_pass(idxs, vals, A, core, xf, batch_nnz, verbose=False, desc=None, loglik=False):
    """Pooled core numerator (ρ, R, ρ) = Σ_k Σ_p w_p · ∂x̂_p/∂C through site k.

    Middle sites: TT's site GEMM  Σ_p outer(w L_k[p], A_k row, R_{k+1}[p]).
    Site 0 sees C through αᵀC, site N-1 through C·ω, so their terms are
    α ⊗ (Σ_p w A_0 row ⊗ R_1) and (Σ_p w L_{N-1} ⊗ A_{N-1} row) ⊗ ω — two
    (b × R)·(b × ρ) GEMMs, as cheap as TT's end sites.

    ``loglik=True`` returns ``(num, Σ x log x̂)``; the sum is free (x̂ is
    already built) and is what the adaptive core step compares.
    """
    C, alpha, omega = core
    n = len(idxs)
    rho, Rk = int(C.shape[0]), int(C.shape[1])
    tt = expand_uniform(core, n)
    num = cp.zeros_like(C)
    ll = cp.zeros((), dtype=cp.float64)

    def _body(start, end):
        mats, L, R, w, xhat = _batch_chain(tt, A, idxs, start, end, vals, xf)
        b = end - start
        part = cp.zeros_like(C)
        for k in range(1, n - 1):
            LW = (L[k] * w[:, None])[:, :, None] * mats[k][:, None, :]      # (b, ρ, R)
            part += (LW.reshape(b, -1).T @ R[k + 1]).reshape(rho, Rk, rho)
        Q0 = (mats[0] * w[:, None]).T @ R[1]                                # (R, ρ)
        QN = (L[n - 1] * w[:, None]).T @ mats[n - 1]                        # (ρ, R)
        part += alpha[:, None, None] * Q0[None, :, :]
        part += QN[:, :, None] * omega[None, None, :]
        ll_b = _xlogxhat(vals[start:end], xhat) if loglik else None
        num[...] += part
        if loglik:
            ll[...] += ll_b

    if int(vals.size):
        _run_nnz_batches(int(vals.size), batch_nnz, _body, verbose, desc)
    return (num, ll) if loglik else num


def boundary_pass(idxs, vals, A, core, xf, batch_nnz, verbose=False, desc=None, loglik=False):
    """Numerators of the α and ω steps plus Σ x, stacked as (2ρ+1,):

        ∂x̂/∂α = S(i_0) R_1   →  Σ_p w_p ∂x̂_p/∂α = Σ_{r,b} C[:, r, b] · Q0[r, b]
        ∂x̂/∂ω = L_{N-1} S(i_{N-1})  →  Σ_{a,r} C[a, r, :] · QN[a, r]

    with Q0, QN the same (R×ρ) GEMMs as the core pass. Every part is a plain
    sum, so shard results simply add. ``loglik=True`` returns
    ``(out, Σ x log x̂)``, as in ``core_pass``."""
    C, alpha, omega = core
    n = len(idxs)
    rho, Rk = int(C.shape[0]), int(C.shape[1])
    tt = expand_uniform(core, n)
    acc = cp.zeros(2 * rho * Rk, dtype=A.dtype)          # [Q0 (R, ρ), QN (ρ, R)]
    out = cp.zeros(2 * rho + 1, dtype=A.dtype)
    ll = cp.zeros((), dtype=cp.float64)
    if not int(vals.size):
        return (out, ll) if loglik else out

    def _body(start, end):
        mats, L, R, w, xhat = _batch_chain(tt, A, idxs, start, end, vals, xf)
        Q0 = (mats[0] * w[:, None]).T @ R[1]
        QN = (L[n - 1] * w[:, None]).T @ mats[n - 1]
        q = cp.concatenate([Q0.ravel(), QN.ravel()])
        ll_b = _xlogxhat(vals[start:end], xhat) if loglik else None
        acc[...] += q
        if loglik:
            ll[...] += ll_b

    _run_nnz_batches(int(vals.size), batch_nnz, _body, verbose, desc)
    Q0 = acc[:rho * Rk].reshape(Rk, rho)
    QN = acc[rho * Rk:].reshape(rho, Rk)
    out[:rho] = (C * Q0[None, :, :]).sum(axis=(1, 2))
    out[rho:2 * rho] = (C * QN[:, :, None]).sum(axis=(0, 1))
    out[2 * rho] = cp.sum(vals)
    return (out, ll) if loglik else out


def error_pass(idxs, vals, A, core, xf, batch_nnz, epsilon, verbose=False, desc=None):
    """(Σ_nnz [x log(x/x̂) − x + x̂], Σ_nnz x̂, Σ_nnz x) as 0-d arrays; the
    same body as tt_kl_compute_errors (left sweep only)."""
    n = len(idxs)
    tt = expand_uniform(core, n)
    x_nz = cp.clip(cp.asarray(vals), a_min=epsilon, a_max=None)
    kl_pos = cp.asarray(0.0, dtype=A.dtype)
    sum_xhat = cp.asarray(0.0, dtype=A.dtype)

    def _body(start, end):
        mats = [A[idxs[k][start:end]] for k in range(n)]
        xhat = cp.clip(left_envs(sites(tt, mats, cp), cp)[n][:, 0], a_min=xf, a_max=None)
        x_b = x_nz[start:end]
        pos = cp.sum(x_b * cp.log(x_b / xhat) - x_b + xhat)
        tot = cp.sum(xhat)
        kl_pos[...] += pos
        sum_xhat[...] += tot

    if int(vals.size):
        _run_nnz_batches(int(vals.size), batch_nnz, _body, verbose, desc)
    return kl_pos, sum_xhat, cp.sum(x_nz)


# ---------------------------------------------------------------------------
# Denominators (closed form) and the steps themselves
# ---------------------------------------------------------------------------

def factor_den(core, A, n_modes, epsilon):
    """Σ_all ∂x̂/∂A[i, r] = Σ_k Ls[k]ᵀ M_r Rs[k+1]  (the same for every i), (R,)."""
    C64, _s, _Ls, _Rs, P = _closed_forms(core, A, n_modes, epsilon)
    den = (C64 * P[:, None, :]).sum(axis=(0, 2))
    return cp.clip(den.astype(A.dtype), a_min=epsilon, a_max=None)


def core_den(core, A, n_modes, epsilon):
    """Σ_all ∂x̂/∂C[a, r, b] = P[a, b] · s[r], (ρ, R, ρ)."""
    _C64, s, _Ls, _Rs, P = _closed_forms(core, A, n_modes, epsilon)
    den = P[:, None, :] * s[None, :, None]
    return cp.clip(den.astype(A.dtype), a_min=epsilon, a_max=None)


def boundary_den(core, A, n_modes, epsilon):
    """(Rs[0], Ls[N]) = (Σ_all ∂x̂/∂α, Σ_all ∂x̂/∂ω), stacked as (2ρ,)."""
    _C64, _s, Ls, Rs, _P = _closed_forms(core, A, n_modes, epsilon)
    den = cp.concatenate([Rs[0], Ls[n_modes]])
    return cp.clip(den.astype(A.dtype), a_min=epsilon, a_max=None)


def apply_factor_step(core, A, num, den, n_modes, epsilon, power=None):
    """A ← A ⊛ (num/den)^power, ℓ1-normalize, scale absorbed into C ONCE
    (every leg then sees A/s · s, so the rescale is exact), then the gauge."""
    ratio = cp.clip(num, a_min=0.0, a_max=None) / den[None, :]
    if power is not None and float(power) != 1.0:
        ratio = ratio ** float(power)
    A_new = cp.clip(A * ratio, a_min=epsilon, a_max=None)
    scale = cp.clip(cp.sum(A_new, axis=0), a_min=epsilon, a_max=None)
    A_new = cp.clip(A_new / scale[None, :], a_min=epsilon, a_max=None)
    core[0] = core[0] * scale[None, :, None]
    normalize_gauge(core, A_new, n_modes)
    return A_new


def apply_core_step(core, A, num, den, n_modes, epsilon, power=None):
    """C ← C ⊛ (num/den)^γ, undamped (γ = 1) unless given, then the gauge."""
    ratio = cp.clip(num, a_min=0.0, a_max=None) / den
    if power is not None and float(power) != 1.0:
        ratio = ratio ** float(power)
    core[0] = cp.clip(core[0] * ratio, a_min=epsilon, a_max=None)
    normalize_gauge(core, A, n_modes)
    return core


def apply_boundary_step(core, A, num, den, n_modes, epsilon, power=None):
    """Joint (α, ω) step, undamped unless ``power`` is given (1/2 is MM-safe:
    x̂ is bilinear in them), then the exact optimal overall scale
    c = Σx / Σ_all x̂ (closed form, monotone), which restores Σ_all x̂ = Σ x.
    ``num`` is the boundary_pass output (2ρ+1,)."""
    rho = int(core[0].shape[0])
    sum_x = float(num[2 * rho])
    ratio = cp.clip(num[:2 * rho], a_min=0.0, a_max=None) / den
    if power is not None and float(power) != 1.0:
        ratio = ratio ** float(power)
    core[1] = cp.clip(core[1] * ratio[:rho], a_min=epsilon, a_max=None)
    core[2] = cp.clip(core[2] * ratio[rho:], a_min=epsilon, a_max=None)
    sum_all = float(utt_sum_all_entries(core, [A], epsilon, n_modes=n_modes))
    if sum_x > 0.0 and math.isfinite(sum_all) and sum_all > 0.0:
        core[1] = core[1] * core[1].dtype.type(sum_x / sum_all)
    normalize_gauge(core, A, n_modes, perron=False)
    return core


# ---------------------------------------------------------------------------
# Adaptive core exponent (``--utt-core-power auto``)
# ---------------------------------------------------------------------------

class AdaptiveCorePower:
    """Exponent γ of the core step, chosen by backtracking.

    Each iteration the core step is judged on that iteration's own NNZ window,
    by the KL at the best overall scale (the boundary step applies that scale
    right after, so a pure scale overshoot is not held against the core):

        J = log Σ_all x̂ − (Σ_nnz x log x̂) / Σ x      (+ a constant)

    in the units of the logged relative error. Before and after are computed
    on the same entries, so the comparison has no sampling noise.

    - J went up: undo the step, halve γ, retry (one extra pass per retry).
    - J did not go up: keep the step; if it was the first try, γ ← γ·``grow``.
    - γ never drops below 1/N (the MM-safe exponent); a step at 1/N is kept.

    γ lies in [1/N, 1] and starts at 1, so a well-behaved run never leaves
    the undamped step. γ lives in this object: it carries across iterations,
    not across a resume (a resumed run starts again at 1).

    TODO (later): the error spikes (around iteration 180 in the 4-gram run) still
    occur with ``auto``; verified 2026-11-12, cause not solved.
    """

    def __init__(self, start=1.0, grow=1.05, shrink=0.5, tol=1e-6, verbose=True):
        self.gamma = float(start)
        self.grow, self.shrink, self.tol = float(grow), float(shrink), float(tol)
        self.verbose = verbose
        self.n_rejected = 0
        self.history = []          # (γ tried, ΔJ, kept) for every try

    def judge(self, delta, n_modes, first_try):
        """Record one tried step with change ``delta`` in J; True = keep it."""
        floor = 1.0 / n_modes
        at_floor = self.gamma <= floor * (1.0 + 1e-9)
        keep = delta <= self.tol or at_floor
        self.history.append((self.gamma, float(delta), keep))
        if keep:
            if first_try:
                self.gamma = min(1.0, self.gamma * self.grow)
            return True
        new = max(floor, self.gamma * self.shrink)
        if self.verbose:
            print(f"  [UTT] core step at γ={self.gamma:.3g} raised the window KL by "
                  f"{delta:.3e}; undone, retrying at γ={new:.3g}")
        self.gamma = new
        self.n_rejected += 1
        return False

    def __repr__(self):
        return f"AdaptiveCorePower(gamma={self.gamma:.3g}, rejected={self.n_rejected})"


def _window_objective(ll, sum_x, core, A, n_modes, epsilon):
    """J of ``AdaptiveCorePower`` from the pass sum ``ll`` = Σ x log x̂."""
    total = float(utt_sum_all_entries(core, [A], epsilon, n_modes=n_modes))
    return math.log(max(total, epsilon)) - float(ll) / sum_x


def core_and_boundary_steps(core, A, shape, epsilon, run_pass, power=None,
                            boundary_power=None):
    """The core step, then the joint (α, ω) step, on whatever ``run_pass`` streams.

    ``run_pass(kind, core, xf, loglik)`` runs the 'core' or 'boundary' NNZ pass
    (single GPU or reduced over shards) and returns its numerator, or
    ``(numerator, Σ x log x̂)`` when ``loglik``. ``power`` is the core exponent:
    None/float (fixed) or an ``AdaptiveCorePower`` (backtracking). ``core`` is
    updated in place.
    """
    n = len(shape)
    den = core_den(core, A, n, epsilon)
    xf = _xhat_floor(core, A, shape, epsilon)

    if not isinstance(power, AdaptiveCorePower):
        num = run_pass("core", core, xf, False)
        apply_core_step(core, A, num, den, n, epsilon, power)
        xf = _xhat_floor(core, A, shape, epsilon)
        bnum = run_pass("boundary", core, xf, False)
    else:
        num, ll0 = run_pass("core", core, xf, True)
        before = [x.copy() for x in core]
        first_try = True
        while True:
            apply_core_step(core, A, num, den, n, epsilon, power.gamma)
            xf = _xhat_floor(core, A, shape, epsilon)
            bnum, ll1 = run_pass("boundary", core, xf, True)
            sum_x = float(bnum[-1])
            if not sum_x > 0.0:
                break
            delta = (_window_objective(ll1, sum_x, core, A, n, epsilon)
                     - _window_objective(ll0, sum_x, before, A, n, epsilon))
            if power.judge(delta, n, first_try):
                break
            core[:] = [x.copy() for x in before]
            first_try = False

    apply_boundary_step(core, A, bnum, boundary_den(core, A, n, epsilon), n, epsilon,
                        boundary_power)
    return core


# ---------------------------------------------------------------------------
# Single-GPU kernels (UpdateRouting contract)
# ---------------------------------------------------------------------------

def utt_kl_factor_update(*args, **kwargs):
    raise NotImplementedError(
        "decomposition='utt' has no per-mode factor update: every mode shares one "
        "factor, which the pooled tied update handles (shared_factors must cover all modes)."
    )


def tied_factor_setup(core, factors, group, shape, epsilon, batch_nnz):
    """Checks and closed-form parts of the tied factor step, shared by the
    single-GPU and sharded updates: ``(A, den, xf, batch_nnz)``."""
    n = len(shape)
    _check_core(core, n)
    if sorted(group) != list(range(n)):
        raise ValueError(f"decomposition='utt' ties all {n} modes; got group {tuple(group)}")
    A = tied_factor(factors)
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_utt(core, factors)
    return A, factor_den(core, A, n, epsilon), _xhat_floor(core, A, shape, epsilon), batch_nnz


def utt_kl_tied_factor_update(vec_tensor, core, factors, group, shape,
                              thread_budget=None, epsilon=1e-12, verbose=False,
                              batch_nnz=None, power=None):
    """Pooled KL update of the shared factor A. ``core`` is updated in place
    (the ℓ1 scale and the gauge go into C, α, ω)."""
    shape = tuple(int(s) for s in shape)
    if verbose:
        print("  [UTT-KL] Updating shared factor...")
    idxs, vals = coo_to_coords(vec_tensor, shape)
    A, den, xf, batch_nnz = tied_factor_setup(core, factors, group, shape, epsilon, batch_nnz)
    num = factor_pass(idxs, vals, A, core, xf, batch_nnz, verbose, "  [UTT-KL] factor")
    return apply_factor_step(core, A, num, den, len(shape), epsilon, power)


def utt_kl_core_update(vec_tensor, shape, core, factors, modes=None,
                       thread_budget=None, epsilon=1e-12, verbose=False,
                       batch_nnz=None, power=None, boundary_power=None):
    """KL update of the shared core C (one pooled pass), then the joint (α, ω)
    step (one more pass). Both undamped unless ``power`` / ``boundary_power``
    are given; ``power`` may be an ``AdaptiveCorePower``. Returns ``core``
    (the same list, updated in place)."""
    shape = tuple(int(s) for s in shape)
    n = len(shape)
    _check_core(core, n)
    A = tied_factor(factors)
    idxs, vals = coo_to_coords(vec_tensor, shape)
    if int(vals.size) == 0:
        return core
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_utt(core, factors)

    if verbose:
        print("  [UTT-KL] Updating shared core...")

    def run_pass(kind, state, xf, loglik):
        fn = core_pass if kind == "core" else boundary_pass
        return fn(idxs, vals, A, state, xf, batch_nnz, verbose, f"  [UTT-KL] {kind}",
                  loglik=loglik)

    return core_and_boundary_steps(core, A, shape, epsilon, run_pass, power, boundary_power)


def utt_kl_compute_errors(vec_tensor, shape, core, factors,
                          thread_budget=None, epsilon=1e-12, verbose=False,
                          batch_nnz=None):
    """Relative generalized KL divergence, normalized as tt_kl_compute_errors:
    D = Σ_nnz [x log(x/x̂) − x + x̂] + (Σ_all x̂ − Σ_nnz x̂),  rel = D / Σ_nnz x."""
    shape = tuple(int(s) for s in shape)
    _check_core(core, len(shape))
    A = factors[0]
    idxs, vals = coo_to_coords(vec_tensor, shape)
    sum_all = utt_sum_all_entries(core, [A], epsilon, n_modes=len(shape))
    if int(vals.size) == 0:
        return cp.asarray(float("nan"), dtype=sum_all.dtype)
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_utt(core, factors)
    xf = _xhat_floor(core, A, shape, epsilon)
    kl_pos, sum_xhat_nz, sum_x = error_pass(idxs, vals, A, core, xf, batch_nnz, epsilon,
                                            verbose, "  [UTT-KL] error x̂ pass")
    return (kl_pos + (sum_all - sum_xhat_nz)) / cp.maximum(sum_x, epsilon)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_uniform_tt(sparse_tensor, shape, rank, modes, init, random_state,
                          tt_rank=100, thread_budget=None, epsilon=1e-12,
                          core_init="random"):
    """Initialize ``([C, alpha, omega], factors)``; every factor is the same matrix.

    The factor comes from TT's own init (``initialize_tucker_tt``: same random /
    SVD handling, same dtype as the data, ℓ1-normalized columns); mode 0's is
    shared by every mode. C ~ U[0.01, 1.01], α, ω ~ U[0.9, 1.1] in that dtype;
    then the gauge, then α and ω scaled so that Σ_all x̂ = Σ x.
    ``core_init="markov"`` builds [C, α, ω] from the data instead
    (experimental/tt_markov_init.py); gauge and rescale are the same.
    """
    from tensormet.tt_hybrid.tt_ops import initialize_tucker_tt

    shape = tuple(int(s) for s in shape)
    n = len(shape)
    rank = [int(r) for r in rank]
    if len(set(rank)) != 1 or len(set(shape)) != 1:
        raise ValueError(f"decomposition='utt' needs one shared dim and rank; got shape "
                         f"{shape}, rank {tuple(rank)}")
    R, rho = rank[0], int(tt_rank)

    if isinstance(init, (tuple, list)) and len(init) == 2:
        compact, factors = init
        return [cp.asarray(x) for x in compact], [cp.asarray(f) for f in factors]
    _tt_cores, factors = initialize_tucker_tt(
        sparse_tensor, shape, rank, modes, init, random_state,
        tt_rank=tt_rank, thread_budget=thread_budget, epsilon=epsilon,
    )
    A = factors[0]
    dtype = A.dtype
    rng = tl.check_random_state(random_state)
    if core_init == "markov":
        from tensormet.experimental.tt_markov_init import markov_utt_core
        t0 = time.perf_counter()
        core = markov_utt_core(sparse_tensor, shape, A, n, rho, rng)
        print(f"[utt-init] markov core from pair statistics ({time.perf_counter() - t0:.1f}s)", flush=True)
    elif core_init == "random":
        core = [cp.asarray(rng.random_sample((rho, R, rho)) + 0.01, dtype=dtype),
                cp.asarray(rng.uniform(0.9, 1.1, rho), dtype=dtype),
                cp.asarray(rng.uniform(0.9, 1.1, rho), dtype=dtype)]
    else:
        raise ValueError(f"core_init must be 'random' or 'markov'; got {core_init!r}")
    normalize_gauge(core, A, n)

    _idxs, vals = coo_to_coords(sparse_tensor, shape)
    sum_x = float(cp.sum(vals))
    sum_all = float(utt_sum_all_entries(core, [A], epsilon, n_modes=n))
    if not (math.isfinite(sum_all) and sum_all > 0.0 and sum_x > 0.0):
        raise FloatingPointError(f"UTT init chain sum is {sum_all!r} (Σx = {sum_x!r}); "
                                 f"cannot rescale. Check rank/tt_rank/shape.")
    c = math.sqrt(sum_x / sum_all)
    core[1] = core[1] * dtype.type(c)
    core[2] = core[2] * dtype.type(c)
    return core, [A] * n


__all__ = [
    "AdaptiveCorePower",
    "core_and_boundary_steps",
    "estimate_batch_nnz_utt",
    "initialize_uniform_tt",
    "normalize_gauge",
    "utt_kl_compute_errors",
    "utt_kl_core_update",
    "utt_kl_factor_update",
    "utt_kl_tied_factor_update",
    "utt_sum_all_entries",
]
