"""
utt_chain.py — chain algebra for the shared-core ("uniform") Tucker-TT.

One core C (ρ, R, ρ) is reused at every position, with boundary vectors α, ω:

    M_r  = C[:, r, :]                    one ρ×ρ operator per latent dimension
    S(i) = Σ_r A[i, r] · M_r             one ρ×ρ operator per word
    x̂(i_0 … i_{N-1}) = αᵀ S(i_0) ⋯ S(i_{N-1}) ω

The compact model is the list ``[C, alpha, omega]``. ``expand_uniform`` turns it
into an ordinary TT core list, so everything in ``tt_hybrid.tt_chain`` (the NNZ
kernels and the eval wrapper alike) runs on it unchanged.

Only broadcasting, ``sum`` and ``@`` are used, so these run on numpy, cupy and
torch alike (``gauge_scales`` takes a host array and returns Python floats).
"""
from __future__ import annotations

import math

import numpy as np


def expand_uniform(compact, n_modes):
    """Standard TT cores [αᵀC (1,R,ρ), C, …, C, C·ω (ρ,R,1)] for an order-N chain.
    The middle entries are the same array C (read-only use only)."""
    C, alpha, omega = compact
    if n_modes == 1:
        G = (alpha[:, None, None] * C * omega[None, None, :]).sum(2).sum(0)
        return [G[None, :, None]]
    first = (alpha[:, None, None] * C).sum(0)[None]             # (1, R, ρ)
    last = (C * omega[None, None, :]).sum(2)[:, :, None]        # (ρ, R, 1)
    return [first] + [C] * (n_modes - 2) + [last]


def transfer_matrix(C, s):
    """T = Σ_r s[r] · M_r (ρ×ρ). With s = factor column sums, T = Σ_i S(i)."""
    return (s[None, :, None] * C).sum(1)


def chain_total(T, alpha, omega, n_modes):
    """αᵀ Tᴺ ω: with T from factor column sums, the sum of x̂ over ALL entries."""
    v = alpha
    for _ in range(n_modes):
        v = v @ T
    return v @ omega


def gauge_scales(T_host, sum_alpha, sum_omega, n_modes):
    """Scalars of the exact gauge reset: ``(lam, alpha_mult, omega_mult)``.

    ``lam`` is the Perron root of ``T_host`` (None when T is not given, or the
    root is not finite and positive): C /= lam and α *= lamᴺ. Then α and ω are
    balanced to equal ℓ1 mass. x̂ is unchanged.
    """
    lam, log_lam = None, 0.0
    if T_host is not None:
        root = float(np.max(np.abs(np.linalg.eigvals(T_host))))
        if math.isfinite(root) and root > 0.0:
            lam, log_lam = root, math.log(root)
    sa, so = float(sum_alpha), float(sum_omega)
    if not (sa > 0.0 and so > 0.0 and math.isfinite(sa) and math.isfinite(so)):
        raise FloatingPointError(f"utt boundary vectors degenerate: Σα={sa!r}, Σω={so!r}")
    # α' = α·λᴺ·c and ω' = ω/c with Σα' = Σω'.
    log_c = 0.5 * (math.log(so) - math.log(sa) - n_modes * log_lam)
    return lam, math.exp(n_modes * log_lam + log_c), math.exp(-log_c)


def boundary_sweeps(T, alpha, omega, n_modes):
    """The column-sum chain: Ls[k] = αᵀ T^k and Rs[k] = T^(N-k) ω, k = 0..N.

    Sums over ALL tensor entries follow from these in closed form:
      Σ_all x̂          = Ls[N]·ω
      factor/core den  = from P = Σ_k outer(Ls[k], Rs[k+1])
      α / ω den        = Rs[0] / Ls[N]
    """
    Ls = [alpha]
    for _ in range(n_modes):
        Ls.append(Ls[-1] @ T)
    Rs = [None] * (n_modes + 1)
    Rs[n_modes] = omega
    for k in range(n_modes - 1, -1, -1):
        Rs[k] = T @ Rs[k + 1]
    return Ls, Rs
