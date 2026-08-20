"""
cp_ops.py — Nonnegative CP (CANDECOMP/PARAFAC) multiplicative-update kernels.

EXPERIMENTAL (see reviews/CP_IMPLEMENTATION_PLAN.md). Everything CP-specific
lives under src/tensormet/experimental/CP/ so the Tucker pipeline is untouched;
the only integration points in the main package are guarded swap points that
default to the Tucker behaviour.

Model (Kolda & Bader 2009, §3):

    X̂ = [[λ; A(1), …, A(N)]] = Σ_{r=1..R} λ_r · a_r(1) ∘ … ∘ a_r(N)

with nonnegative factor matrices A(n) (I_n × R) and weight vector λ (R,).
There is NO core tensor: λ plays the core's role, so these kernels accept the
weight vector through the ``core`` parameter to stay drop-in compatible with
the UpdateRouting seam of ``SparseTupleTensor.non_negative_tucker_with_similarity``.

Representation invariant maintained by the factor updates (CP-APR-style
normalization, Chi & Kolda 2012 §4.2): after each mode's update, that mode's
columns are normalized (ℓ1 for KL, ℓ2 for FR) and the norms are absorbed into
λ. The λ update therefore happens INSIDE the factor update: the ``core``
(weights) array handed in by the loop is updated IN PLACE, and the loop's
"core slot" (``cp_weight_update`` / ``cp_fr_combined_weights_errors``) is a
clip-passthrough (+ fused FR error on log steps).

All kernels follow the ``distance.py`` conventions: CuPy in/out, block-encoded
COO input (``vec_tensor``), ``thread_budget``/``epsilon``/``verbose`` kwargs,
ε-clipping against zero-locking (Lin 2007).

Only ONE kernel family exists (NNZ-streaming): CP has no dense-Z formulation
worth keeping — the streaming form is simultaneously the memory-safe and the
fast path, so the Tucker dense-vs-largedim routing split does not apply.
Transients are O(batch_nnz · R), far smaller than Tucker's R^N objects.

References
----------
[KB09]  Kolda & Bader, SIAM Review 51(3), 2009.       (CP framework, identities)
[LS01]  Lee & Seung, NeurIPS 13, 2001.                (MU rules, monotonicity)
[WW01]  Welling & Weber, PRL 22(12), 2001.            (nonnegative CP MU, FR)
[CK12]  Chi & Kolda, SIMAX 33(4), 2012.               (CP-APR: Poisson/KL MU)
[BK07]  Bader & Kolda, SISC 30(1), 2007.              (sparse MTTKRP, fast fit)
[Lin07] Lin, IEEE TNN 18(6), 2007.                    (ε-safeguarded MU)
[BG08]  Boutsidis & Gallopoulos, PR 41, 2008.         (NNDSVD init)
"""
from __future__ import annotations

import math

import numpy as np
import tensorly as tl
from tqdm import tqdm

from tensormet.utils import make_lazy_cupy_pair
from tensormet.distance import (
    coo_to_coords,
    _gpu_free_bytes,
)

cp, cpx_sparse = make_lazy_cupy_pair()


# ---------------------------------------------------------------------------
# NNZ decode + batching helpers
# ---------------------------------------------------------------------------

def _decode_nnz(vec_tensor, shape):
    """Per-mode coordinate arrays + values for either CuPy storage form.

    Thin alias for the Tucker largedim seam (``distance.coo_to_coords``); the
    arrays are (nnz,) each and live on the device that owns ``vec_tensor``.
    Free for a coordinate-backed tensor, a block decode for the legacy form.
    """
    return coo_to_coords(vec_tensor, tuple(int(s) for s in shape))


def estimate_batch_nnz_cp(factors, safety=0.7, temp_mult=3.0, reserve_b=0):
    """Safe NNZ batch size for the streaming CP kernels.

    Per NNZ entry a batch holds: N gathered factor rows (N·R), the running
    Hadamard product (R), and the SpMM scatter inputs (~2·R with index
    arrays), so ~(N+3)·R·itemsize bytes before elementwise temporaries
    (``temp_mult``). Deliberately much simpler than the Tucker estimators —
    CP transients never contain an R^N object (plan §4, memory note).

    reserve_b :
        Bytes held back for allocations not live at estimate time (the
        kernels' own decode arrays when the estimate is hoisted), mirroring
        the convention of distance.py's estimators.
    """
    N = len(factors)
    R = max(int(f.shape[1]) for f in factors)
    itemsize = int(np.dtype(factors[0].dtype).itemsize)
    bytes_per_p = int(math.ceil((N + 3) * R * itemsize * temp_mult))

    free_b = max(1, int(_gpu_free_bytes()) - int(reserve_b))
    budget_b = int(free_b * safety)
    b = max(1, budget_b // max(1, bytes_per_p))

    # Hard cap proportional to free VRAM, same rail style as distance.py.
    hard_cap = max(1, int(free_b * 0.9 // max(1, bytes_per_p)))
    return min(int(b), hard_cap)


def _gathered_hadamard(factors, idxs, start, end, skip_mode=None):
    """Hadamard product of factor rows gathered at NNZ coordinates.

    Returns (b, R) = ⊛_{m != skip_mode} A(m)[i_m(p), :] for p in [start, end).
    This is the shared gather–Hadamard primitive of both the model-value and
    the MTTKRP kernels [BK07 §5.2].
    """
    H = None
    for m, F in enumerate(factors):
        if m == skip_mode:
            continue
        rows = F[idxs[m][start:end]]  # fancy indexing → fresh array
        if H is None:
            H = rows
        else:
            H *= rows
    return H


# ---------------------------------------------------------------------------
# Shared primitives (plan §1.2/§1.3): model values at NNZ + weighted MTTKRP
# ---------------------------------------------------------------------------

def cp_values_at_nnz(vec_tensor, shape, weights, factors, *,
                     batch_nnz=None, epsilon=1e-12):
    """Model values x̂_p = Σ_r λ_r Π_n A(n)[i_n(p), r] at every NNZ of X.

    Streaming gather–Hadamard–sum over NNZ [BK07 §5]; transients are
    O(batch_nnz · R). Returns a clipped (nnz,) CuPy array.
    """
    idxs, vals = _decode_nnz(vec_tensor, shape)
    return _cp_values_from_idxs(idxs, int(vals.size), weights, factors,
                                batch_nnz=batch_nnz, epsilon=epsilon)


def _cp_values_from_idxs(idxs, nnz, weights, factors, *,
                         batch_nnz=None, epsilon=1e-12):
    xhat = cp.empty(nnz, dtype=factors[0].dtype)
    if nnz == 0:
        return xhat
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_cp(factors)
    for start in range(0, nnz, int(batch_nnz)):
        end = min(start + int(batch_nnz), nnz)
        H = _gathered_hadamard(factors, idxs, start, end)  # (b, R)
        xhat[start:end] = H @ weights
    return cp.clip(xhat, a_min=epsilon, a_max=None)


def cp_weighted_mttkrp(vec_tensor, shape, factors, mode, w=None, *,
                       batch_nnz=None):
    """Sparse (weighted) MTTKRP: M[i_mode(p), :] += w_p · ⊛_{m≠mode} A(m)[i_m(p), :].

    ``w=None`` uses the tensor's own values (the plain FR numerator);
    otherwise ``w`` is an (nnz,) weight vector aligned with the tensor's
    decoded NNZ order (the KL Φ numerator uses w = x/x̂). Scatter-add is done
    via a cuSPARSE SpMM per batch (no serialized atomics), the same skeleton
    as ``kl_factor_update_largedim`` minus the core contraction [BK07; SK15].
    """
    shape = tuple(int(s) for s in shape)
    idxs, vals = _decode_nnz(vec_tensor, shape)
    if w is None:
        w = vals
    out = cp.zeros((shape[mode], factors[mode].shape[1]), dtype=factors[0].dtype)
    _cp_mttkrp_from_idxs(out, idxs, int(vals.size), factors, mode, w,
                         batch_nnz=batch_nnz)
    return out


def _cp_mttkrp_from_idxs(out, idxs, nnz, factors, mode, w, *, batch_nnz=None):
    """Accumulate the weighted MTTKRP into ``out`` (I_mode, R), in place."""
    if nnz == 0:
        return out
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_cp(factors)
    for start in range(0, nnz, int(batch_nnz)):
        end = min(start + int(batch_nnz), nnz)
        H = _gathered_hadamard(factors, idxs, start, end, skip_mode=mode)  # (b, R)
        b = end - start
        row_idx_b = idxs[mode][start:end].astype(cp.int32)
        col_idx_b = cp.arange(b, dtype=cp.int32)
        S_b = cpx_sparse.csr_matrix(
            (w[start:end], (row_idx_b, col_idx_b)),
            shape=(out.shape[0], b),
        )
        out += S_b @ H
    return out


def _cp_kl_phi_from_idxs(out, B, idxs, vals, nnz, factors, mode, *,
                         batch_nnz=None, epsilon=1e-12, verbose=False):
    """Accumulate the KL numerator Φ into ``out`` (I_mode, R), in place.

    Φ = ( X_(mode) ⊘ (B Π) ) Πᵀ over the NNZ only. One fused pass: the model
    value x̂, the ratio weight x/x̂ and the scatter all share the same gathered
    Hadamard rows. Depends on ``B``, so it must be recomputed per inner
    iteration (unlike the FR MTTKRP, whose weights are the tensor's values).
    """
    if nnz == 0:
        return out
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_cp(factors)
    rows_mode = idxs[mode]
    batches = range(0, nnz, int(batch_nnz))
    if verbose:
        batches = tqdm(batches, desc=f"  [CP-KL] factor {mode} nnz-batches",
                       unit="batch", leave=False)
    for start in batches:
        end = min(start + int(batch_nnz), nnz)
        H = _gathered_hadamard(factors, idxs, start, end, skip_mode=mode)  # (b, R)
        r_b = rows_mode[start:end]
        xhat_b = cp.sum(B[r_b] * H, axis=1)               # x̂ at NNZ
        xhat_b = cp.clip(xhat_b, a_min=epsilon, a_max=None)
        w_b = vals[start:end] / xhat_b                     # x / x̂

        b = end - start
        S_b = cpx_sparse.csr_matrix(
            (w_b, (r_b.astype(cp.int32), cp.arange(b, dtype=cp.int32))),
            shape=(out.shape[0], b),
        )
        out += S_b @ H
    return out


def _hadamard_of_grams(factors, skip_mode=None, epsilon=1e-12):
    """Γ = ⊛_{m≠skip_mode} A(m)ᵀ A(m)  — (R, R), identity (i) of [KB09 §2.6]."""
    G = None
    for m, F in enumerate(factors):
        if m == skip_mode:
            continue
        g = F.T @ F
        G = g if G is None else G * g
    return cp.clip(G, a_min=epsilon, a_max=None)


def _colsum_products(factors, skip_mode=None, epsilon=1e-12):
    """σ_r = Π_{m≠skip_mode} 1ᵀ a_r(m)  — (R,), identity (ii) of [KB09 §2.6]."""
    s = None
    for m, F in enumerate(factors):
        if m == skip_mode:
            continue
        c = cp.sum(F, axis=0)
        s = c if s is None else s * c
    return cp.clip(s, a_min=epsilon, a_max=None)


# ---------------------------------------------------------------------------
# Primary-side algebra: MU steps and the normalize/absorb tail
# ---------------------------------------------------------------------------
# Split out from the factor updates so the NNZ-dependent accumulation (the
# MTTKRP / Φ kernels above) and this NNZ-free remainder can be driven
# separately — the single-GPU kernels below just compose the two.

def _cp_fr_mu_step(B, M, Gamma, epsilon=1e-12):
    """B ← B ⊛ M ⊘ (B Γ), ε-clipped. Γ is NNZ-free (Hadamard of Grams)."""
    denominator = cp.clip(B @ Gamma, a_min=epsilon, a_max=None)
    B = B * (M / (denominator + 1e-12))
    return cp.clip(B, a_min=epsilon, a_max=None)


def _cp_kl_mu_step(B, Phi, sigma, epsilon=1e-12, scooch_kappa=0.0):
    """B ← B ⊛ Φ ⊘ (1 σᵀ), ε-clipped. σ is NNZ-free (column-sum products).

    ``scooch_kappa`` > 0 nudges near-zero entries whose multiplier exceeds 1
    ("inadmissible zeros") so they can re-enter the support [CK12 §4.1].
    """
    if scooch_kappa > 0.0:
        inadmissible = (B <= 2.0 * epsilon) & ((Phi / sigma[None, :]) > 1.0)
        B = cp.where(inadmissible, B + scooch_kappa, B)
    B = B * (Phi / sigma[None, :])
    return cp.clip(B, a_min=epsilon, a_max=None)


def _cp_absorb_into_weights(B, weights, norm="l2", epsilon=1e-12):
    """Normalize B's columns and absorb the norms into λ (cp_normalize
    semantics). ``weights`` is updated IN PLACE — that is how the loop's
    ``core`` variable stays current. Returns the renormalized factor.
    """
    lam = cp.sum(B, axis=0) if norm == "l1" else cp.linalg.norm(B, axis=0)
    lam = cp.clip(lam, a_min=epsilon, a_max=None)
    A_new = cp.clip(B / lam[None, :], a_min=epsilon, a_max=None)
    weights[...] = lam
    return A_new


# ---------------------------------------------------------------------------
# FR (Frobenius / least-squares) factor update  — plan §1.2 [WW01; LS01 Thm.1]
# ---------------------------------------------------------------------------

def cp_fr_factor_update(vec_tensor, core, factors, mode, shape,
                        thread_budget=None, epsilon=1e-12, verbose=False,
                        batch_nnz=None):
    """One FR multiplicative update for CP factor A(mode).

        B = A(mode) diag(λ)
        B ← B ⊛ MTTKRP(X, {A}, mode) ⊘ (B · Γ),   Γ = ⊛_{m≠mode} A(m)ᵀA(m)
        λ ← ‖b_r‖₂ ;  A(mode) ← B diag(λ)⁻¹      (cp_normalize semantics)

    ``core`` is the λ weight vector (R,) — it is updated IN PLACE so the
    training loop's ``core`` variable stays current without a loop change.
    Monotone non-increase of ‖X−X̂‖² per block update follows from [LS01];
    the ε-clip is Lin's modified MU [Lin07].
    """
    if verbose:
        print(f"  [CP-FR] Updating factor {mode}...")
    weights = core  # (R,) λ — named `core` for UpdateRouting seam compatibility

    shape = tuple(int(s) for s in shape)
    idxs, vals = _decode_nnz(vec_tensor, shape)
    nnz = int(vals.size)

    A = factors[mode]
    B = A * weights[None, :]

    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_cp(factors)

    # Numerator: MTTKRP with the tensor's values (Z never materialized).
    M = cp.zeros_like(A)
    _cp_mttkrp_from_idxs(M, idxs, nnz, factors, mode, vals, batch_nnz=batch_nnz)

    # Denominator: (I_mode, R) via the tiny R×R Hadamard-of-Grams — never dense.
    Gamma = _hadamard_of_grams(factors, skip_mode=mode, epsilon=epsilon)
    B = _cp_fr_mu_step(B, M, Gamma, epsilon=epsilon)
    return _cp_absorb_into_weights(B, weights, norm="l2", epsilon=epsilon)


# ---------------------------------------------------------------------------
# KL / Poisson factor update (CP-APR MU step) — plan §1.3 [CK12 Alg.3; LS01 Thm.2]
# ---------------------------------------------------------------------------

def cp_kl_factor_update(vec_tensor, core, factors, mode, shape,
                        thread_budget=None, epsilon=1e-12, verbose=False,
                        batch_nnz=None, inner_iters=1, scooch_kappa=0.0):
    """One KL (generalized KL = Poisson NLL) multiplicative update for A(mode).

    With B = A(mode) diag(λ) and Π = Z(mode)ᵀ (never materialized):

        Φ = ( X_(mode) ⊘ (B Π) ) Πᵀ          # only at the NNZ of X
        B ← B ⊛ Φ ⊘ ( 1 σᵀ ),  σ_r = Π_{m≠mode} 1ᵀ a_r(m)
        λ ← 1ᵀ b_r ;  A(mode) ← B diag(λ)⁻¹  (ℓ1, keeps other modes column-stochastic)

    σ is computed explicitly (cheap, O(Σ I_n R)) rather than assumed 1, so the
    update is exact even before the first normalization sweep; once every
    other factor is column-stochastic σ ≈ 1 and the divide is a no-op in
    effect [CK12 §4.2].

    inner_iters : repeat the Φ/B step up to this many times per mode
        (CP-APR's ``maxinner``; default 1 = plain sweep, matching the Tucker
        loop's structure).
    scooch_kappa : if > 0, entries with b ≈ 0 whose multiplier Φ/σ > 1
        ("inadmissible zeros") are nudged up by κ before the multiplicative
        step [CK12 §4.1]. The ε-clip already prevents exact zeros, so this is
        off by default and kept only for fidelity to CP-APR.

    ``core`` is the λ weight vector (R,), updated IN PLACE (see module doc).
    """
    if verbose:
        print(f"  [CP-KL] Updating factor {mode}...")
    weights = core  # (R,) λ

    shape = tuple(int(s) for s in shape)
    idxs, vals = _decode_nnz(vec_tensor, shape)
    nnz = int(vals.size)

    A = factors[mode]
    B = A * weights[None, :]

    sigma = _colsum_products(factors, skip_mode=mode, epsilon=epsilon)  # (R,)

    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_cp(factors)

    for _inner in range(max(1, int(inner_iters))):
        Phi = cp.zeros_like(B)
        _cp_kl_phi_from_idxs(Phi, B, idxs, vals, nnz, factors, mode,
                             batch_nnz=batch_nnz, epsilon=epsilon, verbose=verbose)
        B = _cp_kl_mu_step(B, Phi, sigma, epsilon=epsilon,
                           scooch_kappa=scooch_kappa)

    # ℓ1 for KL: keeps the other modes column-stochastic.
    return _cp_absorb_into_weights(B, weights, norm="l1", epsilon=epsilon)


# ---------------------------------------------------------------------------
# "Core slot" callables: λ passthrough (+ fused FR error on log steps)
# ---------------------------------------------------------------------------

def cp_weight_update(vec_tensor, shape, core, factors, modes=None,
                     thread_budget=None, epsilon=1e-12, verbose=False):
    """λ "core slot" passthrough.

    λ is already current — the factor updates absorbed the column norms in
    place — so this only re-applies the ε floor. Signature mirrors the Tucker
    core updates so UpdateRouting treats both families alike.
    """
    return cp.clip(core, a_min=epsilon, a_max=None)


def cp_fr_combined_weights_errors(vec_tensor, shape, core, factors, modes=None,
                                  thread_budget=None, epsilon=1e-12,
                                  verbose=False, batch_nnz=None):
    """λ passthrough + fused FR relative error, matching the Tucker FR
    ``core_returns_error=True`` contract on log steps.

    Unlike Tucker's ``fr_combined_core_errors`` there is no core MU step to
    fuse — λ is already current — so this is the exact full-tensor error
    (plan §1.4) computed fresh (no reuse of a possibly-subsampled sweep
    MTTKRP; exactness on log steps is worth one extra O(nnz·R) pass).
    """
    weights = cp.clip(core, a_min=epsilon, a_max=None)
    rel_err = cp_fr_compute_errors(
        vec_tensor, shape, weights, factors,
        thread_budget=thread_budget, epsilon=epsilon, verbose=verbose,
        batch_nnz=batch_nnz,
    )
    return weights, rel_err


# ---------------------------------------------------------------------------
# Error / fitness kernels — plan §1.4 (no dense reconstruction, ever)
# ---------------------------------------------------------------------------

def cp_fr_compute_errors(vec_tensor, shape, core, factors,
                         thread_budget=None, epsilon=1e-12, verbose=False,
                         batch_nnz=None):
    """Relative Frobenius error ‖X − X̂‖ / ‖X‖ for sparse X, fast-fit form
    [BK07; KB09 §3.4]:

        ‖X − X̂‖² = ‖X‖² − 2⟨X, X̂⟩ + ‖X̂‖²
        ⟨X, X̂⟩   = Σ_p x_p · x̂_p                    (x̂_p streamed at NNZ)
        ‖X̂‖²     = λᵀ ( ⊛_n A(n)ᵀA(n) ) λ            (all-R×R, O(Σ I_n R²))
    """
    if verbose:
        print("  [CP-FR] Computing Frobenius errors...")
    weights = core

    shape = tuple(int(s) for s in shape)
    idxs, vals = _decode_nnz(vec_tensor, shape)
    nnz = int(vals.size)

    x_nz = cp.clip(vals.astype(factors[0].dtype), a_min=0.0, a_max=None)
    norm_X_sq = cp.sum(x_nz * x_nz)
    norm_X = cp.sqrt(cp.maximum(norm_X_sq, epsilon))

    # ‖X̂‖² in rank space — never dense.
    G_all = _hadamard_of_grams(factors, epsilon=epsilon)  # (R, R)
    norm_Xhat_sq = weights @ (G_all @ weights)

    if nnz == 0:
        return cp.sqrt(cp.maximum(norm_Xhat_sq, 0.0)) / norm_X

    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_cp(factors)

    inner_prod = cp.asarray(0.0, dtype=factors[0].dtype)
    batches = range(0, nnz, int(batch_nnz))
    if verbose:
        batches = tqdm(batches, desc="  [CP-FR] error x̂ pass", unit="batch", leave=False)
    for start in batches:
        end = min(start + int(batch_nnz), nnz)
        H = _gathered_hadamard(factors, idxs, start, end)  # (b, R), all modes
        xhat_b = H @ weights
        inner_prod += cp.sum(x_nz[start:end] * xhat_b)

    residual_sq = cp.maximum(norm_X_sq + norm_Xhat_sq - 2.0 * inner_prod, 0.0)
    return cp.sqrt(residual_sq) / norm_X


def cp_kl_compute_errors(vec_tensor, shape, core, factors,
                         thread_budget=None, epsilon=1e-12, verbose=False,
                         batch_nnz=None):
    """Relative generalized KL divergence D(X‖X̂) / Σ_p x_p for sparse X.

        D = Σ_{nnz} [x log(x/x̂) − x + x̂]  +  ( Σ_all x̂ − Σ_{nnz} x̂ )
        Σ_all x̂ = Σ_r λ_r Π_n (1ᵀ a_r(n))            (identity (ii), closed form)

    Contrast with Tucker's ``kl_compute_errors``, which needs a full dense
    reconstruction on CPU for the zero-entry term: for CP the zero term is a
    closed form, so this path is strictly cheaper and needs no
    ThreadBudget-limited CPU excursion (plan §1.4).
    """
    if verbose:
        print("  [CP-KL] Computing KL errors...")
    weights = core

    shape = tuple(int(s) for s in shape)
    idxs, vals = _decode_nnz(vec_tensor, shape)
    nnz = int(vals.size)

    # Σ over ALL entries of X̂ — exact, closed form, O(Σ I_n R).
    sum_all = cp.sum(weights * _colsum_products(factors, epsilon=epsilon))

    if nnz == 0:
        # All-zero X: KL reduces to Σ x̂ (mirror the Tucker edge-case style).
        return sum_all / cp.maximum(cp.asarray(0.0, dtype=sum_all.dtype), epsilon)

    x_nz = cp.clip(cp.asarray(vals), a_min=epsilon, a_max=None)

    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_cp(factors)

    kl_pos = cp.asarray(0.0, dtype=factors[0].dtype)
    sum_xhat_nz = cp.asarray(0.0, dtype=factors[0].dtype)
    batches = range(0, nnz, int(batch_nnz))
    if verbose:
        batches = tqdm(batches, desc="  [CP-KL] error x̂ pass", unit="batch", leave=False)
    for start in batches:
        end = min(start + int(batch_nnz), nnz)
        H = _gathered_hadamard(factors, idxs, start, end)  # (b, R)
        xhat_b = cp.clip(H @ weights, a_min=epsilon, a_max=None)
        x_b = x_nz[start:end]
        kl_pos += cp.sum(x_b * cp.log(x_b / xhat_b) - x_b + xhat_b)
        sum_xhat_nz += cp.sum(xhat_b)

    kl_zero = sum_all - sum_xhat_nz
    kl_total = kl_pos + kl_zero
    sum_X = cp.sum(x_nz)
    return kl_total / cp.maximum(sum_X, epsilon)


# ---------------------------------------------------------------------------
# Initialization — plan §1.5
# ---------------------------------------------------------------------------

def cp_normalize_absorb(weights, factors, norm="l2", epsilon=1e-12):
    """Normalize every factor's columns and absorb the norms into λ
    (``tensorly.cp_tensor.cp_normalize`` semantics, [KB09 §3.4]).

    norm : "l2" (FR convention) or "l1" (KL / CP-APR column-stochastic).
    Returns (weights, factors) — new arrays, inputs untouched.
    """
    weights = cp.clip(cp.asarray(weights).copy(), a_min=epsilon, a_max=None)
    out_factors = []
    for F in factors:
        if norm == "l1":
            s = cp.sum(F, axis=0)
        elif norm == "l2":
            s = cp.linalg.norm(F, axis=0)
        else:
            raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")
        s = cp.clip(s, a_min=epsilon, a_max=None)
        out_factors.append(cp.clip(F / s[None, :], a_min=epsilon, a_max=None))
        weights = weights * s
    return weights, out_factors


def initialize_nonnegative_cp(sparse_tensor, shape, rank, modes, init,
                              random_state, thread_budget=None,
                              divergence="fr", epsilon=1e-12):
    """Initialize (weights, factors) for nonnegative CP.

    init :
        "random"                 — A(n) ~ U(0,1), λ = 1 (CP-APR default [CK12 §6]).
        "svd"/"svd_cpu"/"svd_loose"/"randomised_svd"/"svd_gpu" — per-mode
            leading-R left singular vectors of
            X_(n) made nonnegative via NNDSVD [BG08], reusing the Tucker init
            machinery in ``sparse_ops`` verbatim and DISCARDING its core (the
            core is one extra streamed NNZ pass — negligible next to the SVDs
            themselves, and reusing the battle-tested routine beats forking it
            for an experimental path).
        (weights, factors) tuple — passthrough.

    The result is normalized with the divergence's convention (ℓ1 for kl,
    ℓ2 for fr) so the kernels' representation invariant holds from iteration 0.

    Returns
    -------
    (weights, factors) : ((R,) CuPy array, list of (I_n, R) CuPy arrays)
    """
    rank = int(rank)
    if init == "random":
        rng = tl.check_random_state(random_state)
        factors = [
            tl.tensor(rng.random_sample((shape[mode], rank)),
                      **tl.context(sparse_tensor))
            for mode in modes
        ]
        factors = [cp.clip(cp.abs(f), a_min=1e-30, a_max=None) for f in factors]
        weights = cp.ones(rank, dtype=factors[0].dtype)
    elif isinstance(init, str) and "svd" in init:
        from tensormet.sparse_ops import initialize_nonnegative_tucker
        _core, factors = initialize_nonnegative_tucker(
            sparse_tensor, shape, [rank] * len(modes), modes, init,
            random_state, thread_budget=thread_budget,
        )
        del _core  # CP has no core; λ absorbs the NNDSVD column scales below
        weights = cp.ones(rank, dtype=factors[0].dtype)
    elif isinstance(init, (tuple, list)) and len(init) == 2:
        weights, factors = init
        weights = cp.asarray(weights)
        factors = [cp.asarray(f) for f in factors]
    else:
        raise ValueError(
            f"CP init must be 'random', 'svd', 'svd_cpu', 'svd_loose', "
            f"'randomised_svd', 'svd_gpu' or a (weights, factors) tuple; got {init!r}"
        )

    norm = "l1" if divergence == "kl" else "l2"
    return cp_normalize_absorb(weights, factors, norm=norm, epsilon=epsilon)
