"""
tt_chain.py — the tensor-train chain algebra shared by the kernels and the
inference wrapper.

The Tucker-TT hybrid keeps the Tucker factor matrices and stores the R^N core
as a tensor train over the rank indices:

    G[r_0 … r_{N-1}] = C_0[:, r_0, :] · C_1[:, r_1, :] · … · C_{N-1}[:, r_{N-1}, :]
    X̂[i_0 … i_{N-1}] = Σ_r G[r] · Π_n A_n[i_n, r_n]

with C_k of shape (ρ_k, R_k, ρ_{k+1}) and ρ_0 = ρ_N = 1.

Every quantity the MU kernels need is a contraction of that chain against one
gathered latent vector per mode, so all of them are built from three steps:

    site   S_k[p] = Σ_r a_k[p, r] · C_k[:, r, :]     (b, ρ_k, ρ_{k+1})
    sweeps L_k    = L_{k-1} S_{k-1},  R_k = S_k R_{k+1}
    grad   Z_k[p, r] = L_k[p] · C_k[:, r, :] · R_{k+1}[p]

Functions take an ``xp`` module so the same code runs on CuPy (kernels), NumPy
and torch (inference); only two-operand einsums are used, which those three
back-ends spell identically.
"""
from __future__ import annotations


def bond_dims(ranks, tt_rank):
    """Bond dimensions ρ_0..ρ_N, capped by ``tt_rank`` and by the exact TT rank
    at each cut (min of the rank-products on either side)."""
    n = len(ranks)
    pre, suf = [1] * (n + 1), [1] * (n + 1)
    for k in range(n):
        pre[k + 1] = pre[k] * ranks[k]
    for k in range(n - 1, -1, -1):
        suf[k] = suf[k + 1] * ranks[k]
    return [min(int(tt_rank), pre[k], suf[k]) for k in range(n + 1)]


def core_shapes(ranks, tt_rank):
    """Shapes of the TT cores for the given Tucker ranks and bond cap."""
    rho = bond_dims(ranks, tt_rank)
    return [(rho[k], int(ranks[k]), rho[k + 1]) for k in range(len(ranks))]


def sites(tt_cores, mats, xp, skip=None):
    """Site matrices S_k = Σ_r mats[k][:, r] · C_k[:, r, :], each (b, ρ_k, ρ_{k+1}).

    ``skip`` leaves that site as None, which is what an "excluded role" query
    needs: the sweeps then stop either side of it and never touch mats[skip].
    """
    return [None if k == skip else xp.einsum("pr,arb->pab", mats[k], C)
            for k, C in enumerate(tt_cores)]


def left_envs(S, xp):
    """L[0..k], where L[k] is (b, ρ_k) — the chain left of site k. Stops at the
    first None site, so L[skip] is the last entry when one site was skipped."""
    ref = next(s for s in S if s is not None)
    L = [xp.ones_like(ref[:, :1, 0])]
    for S_k in S:
        if S_k is None:
            break
        L.append(xp.matmul(L[-1][:, None, :], S_k)[:, 0, :])
    return L


def right_envs(S, xp):
    """R[0..N] with R[k] = (b, ρ_k) the chain right of (and including) site k;
    entries left of a None site stay None."""
    n = len(S)
    ref = next(s for s in S if s is not None)
    R = [None] * (n + 1)
    R[n] = xp.ones_like(ref[:, :1, 0])
    for k in range(n - 1, -1, -1):
        if S[k] is None:
            break
        R[k] = xp.matmul(S[k], R[k + 1][:, :, None])[:, :, 0]
    return R


def site_grad(L_k, C_k, R_k1, xp):
    """(b, R_k) = ∂x̂/∂(latent at site k): Σ_{a,b} L_k[p, a] C_k[a, r, b] R_{k+1}[p, b].

    With L/R built from real latents this is the Tucker "Z" vector of mode k;
    with them built from factor column sums it is the exact MU denominator
    (the sum over ALL tensor entries).
    """
    T = xp.einsum("pa,arb->prb", L_k, C_k)
    return xp.matmul(T, R_k1[:, :, None])[:, :, 0]


def chain_values(tt_cores, mats, xp):
    """Model values x̂ for a batch of entries, (b,)."""
    return left_envs(sites(tt_cores, mats, xp), xp)[len(tt_cores)][:, 0]


def contract(tt_cores, latents, open_positions, xp):
    """Contract the chain against ``latents``, leaving the rank legs at
    ``open_positions`` open. Returns an array with one axis per open position.

    Zero open positions gives the scalar core score, one gives the
    excluded/included role vector, two gives the (R_a, R_b) matrix
    ``get_top_combinations`` ranks over. Latents at open positions are ignored.
    """
    open_positions = sorted(open_positions)
    E = xp.ones_like(tt_cores[0][:1, 0, :1])  # (1, ρ_0) — leading axis is the open block
    dims = []
    for k, C in enumerate(tt_cores):
        if k in open_positions:
            E = xp.tensordot(E, C, axes=([1], [0]))       # (K, R_k, ρ_{k+1})
            E = E.reshape(-1, C.shape[2])
            dims.append(int(C.shape[1]))
        else:
            E = E @ xp.tensordot(latents[k], C, axes=([0], [1]))
    return E.reshape(tuple(dims)) if dims else E.reshape(())


def to_dense_core(tt_cores, xp):
    """Materialize the R^N Tucker core. O(Π R_k) memory — callers guard the size."""
    G = tt_cores[0][0]                                     # (R_0, ρ_1)
    for C in tt_cores[1:]:
        G = xp.tensordot(G, C, axes=([-1], [0]))           # (…, R_k, ρ_{k+1})
    return G[..., 0]
