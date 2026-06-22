from __future__ import annotations

import os
import numpy as np
import torch
from typing import List, Optional, Union, Tuple, Literal
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from tensormet.tucker_tensor import TuckerDecomposition, _to_np, _voc_list_key
from tensormet.utils import voc_index, ThreadBudget
from tensormet.similarity import get_eval_num_threads



class ExtendedTucker(TuckerDecomposition):
    def __init__(
        self,
        core,
        factors: List[torch.Tensor],
        vocab: dict,
        shared_factors: set | None = None,
        roles: Optional[List[str]] = None,
    ):
        super().__init__(core, factors, vocab, shared_factors=shared_factors, roles=roles)

        self.is_extended: bool = False
        self.extended_roles: set[str] = set()

        self.extended_tokens: dict[str, set[str]] = {
            role: set() for role in self.roles
        }

        self.extensions: dict[str, dict[str, np.ndarray]] = {
            role: {} for role in self.roles
        }

        self.extension_counts: dict[str, dict[str, int]] = {
            role: {} for role in self.roles
        }

        self.extension_lengths: dict[str, int] = {
            role: 0 for role in self.roles
        }

    @classmethod
    def from_tucker(cls, t: TuckerDecomposition) -> "ExtendedTucker":
        """
        Create an ExtendedTucker that shares core/factors/vocab references with `t`.
        """
        return cls(
            t.core,
            t.factors,
            t.vocab,
            shared_factors=t.shared_factors,
            roles=t.roles,
        )

    @classmethod
    def extend_tucker(
        cls,
        t: TuckerDecomposition,
        dataset,  # iterable of tuples
        roles: List[str],
        normalize: bool = True,
        normalize_mode: Literal["l2", "minmax"] = "l2",
        n_threads: int | None = None,
        thread_budget: ThreadBudget | None = None,
        fraction_threads: float = 0.75,
        min_threads: int = 1,
        min_count: int | None = None,
        top_k: int | None = None,
    ) -> "ExtendedTucker":
        ext = cls.from_tucker(t)
        for role in roles:
            ext.extend_role(
                role=role,
                sample=dataset,
                normalize=normalize,
                normalize_mode=normalize_mode,
                n_threads=n_threads,
                thread_budget=thread_budget,
                fraction_threads=fraction_threads,
                min_threads=min_threads,
                min_count=min_count,
                top_k=top_k,
            )
        return ext

    def _validate_role(self, role: str) -> None:
        if role not in self.roles:
            raise ValueError(f"role must be one of {set(self.roles)}, got {role!r}")

    def _sync_extension_flags(self) -> None:
        self.extended_roles = {
            role for role in self.roles if self.extension_lengths[role] > 0
        }
        self.is_extended = len(self.extended_roles) > 0

    def check_vocab(self, triple: Tuple[str, ...], return_type=bool) -> bool | tuple:
        """
        True if each element is either in base vocab OR in extensions for that role.
        Mirrors the generalized TuckerDecomposition.check_vocab signature.
        """
        if len(triple) != len(self.roles):
            raise ValueError(
                f"Expected tuple of length {len(self.roles)}, got {len(triple)}"
            )

        in_roles = [
            (triple[i] in self.vocab[voc_index(self.roles[i])]) or
            (triple[i] in self.extensions[self.roles[i]])
            for i in range(len(self.roles))
        ]
        if return_type == tuple:
            return tuple(in_roles)
        return all(in_roles)

    def fetch_single_latent(self, element, role) -> np.ndarray:
        """
        First try base vocab, else fall back to extension dict.
        """
        self._validate_role(role)

        vocab_key = voc_index(role)
        if element in self.vocab[vocab_key]:
            el_idx = self.vocab[vocab_key][element]
            factor_slice = self.factors[self.get_role_index(role)][el_idx]
            return _to_np(factor_slice)

        if element in self.extensions[role]:
            return np.asarray(self.extensions[role][element])

        raise KeyError(f"{element!r} not in base vocab and not extended for role {role!r}")

    # NOTE:
    # fetch_latents does not need to be overridden here.
    # The generalized parent implementation already maps over self.roles and
    # uses fetch_single_latent, so extended tokens work automatically.

    def extend_role(
        self,
        role: str,
        sample,  # iterable of tuples
        normalize: bool = True,
        normalize_mode: Literal["l2", "minmax"] = "l2",
        n_threads: int | None = None,
        thread_budget: ThreadBudget | None = None,
        fraction_threads: float = 0.75,
        min_threads: int = 1,
        min_count: int | None = None,
        top_k: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Extend one role by building representations for OOV tokens from contexts
        where all other roles are in-vocab.
        """
        self._validate_role(role)

        if n_threads is None:
            n_threads = get_eval_num_threads(
                fraction=fraction_threads,
                min_threads=min_threads,
            )

        r_idx = self.get_role_index(role)
        other_idxs = [i for i in range(len(self.roles)) if i != r_idx]

        this_vocab = self.vocab[voc_index(role)]
        other_roles = [self.roles[i] for i in other_idxs]
        other_vocabs = [self.vocab[voc_index(r)] for r in other_roles]

        eps = 1e-12
        range_q = (1.0, 99.0)

        if normalize and normalize_mode == "l2":
            F_base = _to_np(self.factors[r_idx]).astype(np.float64, copy=False)
            base_row_norms = np.linalg.norm(F_base, axis=1)
            nz = base_row_norms[np.isfinite(base_row_norms) & (base_row_norms > 0)]
            target_norm = float(np.median(nz)) if nz.size else 1.0

            def _post_normalize(extension: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                out2 = {}
                for tok, vec in extension.items():
                    vec = np.asarray(vec, dtype=np.float64)
                    n = float(np.linalg.norm(vec))
                    if (not np.isfinite(n)) or (n < eps):
                        out2[tok] = np.zeros_like(vec, dtype=np.float64)
                    else:
                        out2[tok] = vec * (target_norm / (n + eps))
                return out2

        elif normalize and normalize_mode == "minmax":
            F_base = _to_np(self.factors[r_idx]).astype(np.float64, copy=False)

            lo_q, hi_q = range_q
            base_lo = np.nanpercentile(F_base, lo_q, axis=0)
            base_hi = np.nanpercentile(F_base, hi_q, axis=0)
            base_span = np.maximum(base_hi - base_lo, eps)
            base_mid = (base_lo + base_hi) * 0.5

            def _post_normalize(extension: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                if not extension:
                    return extension

                toks = list(extension.keys())
                E = np.stack(
                    [np.asarray(extension[t], dtype=np.float64) for t in toks],
                    axis=0,
                )

                ext_lo = np.nanpercentile(E, lo_q, axis=0)
                ext_hi = np.nanpercentile(E, hi_q, axis=0)
                ext_span = ext_hi - ext_lo

                flat = ext_span < eps
                safe_span = np.where(flat, 1.0, ext_span)

                E2 = (E - ext_lo) * (base_span / safe_span) + base_lo
                E2[:, flat] = base_mid[flat]

                bad = ~np.isfinite(E2)
                if np.any(bad):
                    E2[bad] = np.take(base_mid, np.where(bad)[1])

                return {t: E2[i] for i, t in enumerate(toks)}

        else:
            def _post_normalize(extension: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                return extension

        # collect OOV contexts
        out = defaultdict(list)
        for tpl in tqdm(sample, desc=f"building OOV add list ({role})"):
            if len(tpl) != len(self.roles):
                raise ValueError(
                    f"Sample tuple has length {len(tpl)} but expected {len(self.roles)}"
                )

            tok = tpl[r_idx]

            if (tok in this_vocab) or (tok in self.extensions[role]):
                continue

            if all(tpl[o_i] in o_vocab for o_i, o_vocab in zip(other_idxs, other_vocabs)):
                out[tok].append(tpl)

        if not out:
            return {}

        # filter by min_count / top_k
        if (min_count is not None) or (top_k is not None):
            counts0 = {tok: len(ctxs) for tok, ctxs in out.items()}

            if min_count is not None:
                keep = {tok for tok, c in counts0.items() if c >= min_count}
            else:
                keep = set(counts0.keys())

            if top_k is not None:
                ranked = sorted(keep, key=lambda t: (-counts0[t], t))
                keep = set(ranked[:top_k])

            out = {tok: out[tok] for tok in keep}

        if not out:
            return {}

        limiter = thread_budget.limit() if thread_budget is not None else None
        if limiter is None:
            ctx = contextmanager(lambda: (yield))()
        else:
            ctx = limiter

        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}

        def _one_call(tok, ctx_tuple):
            rep = self.excluded_role_vector(ctx_tuple, role)
            rep = np.asarray(rep, dtype=np.float64)
            return tok, rep

        jobs = [(tok, ctx_tuple) for tok, ctxs in out.items() for ctx_tuple in ctxs]
        if not jobs:
            return {}

        with ctx:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                futures = [ex.submit(_one_call, tok, ctx_tuple) for tok, ctx_tuple in jobs]

                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"calculating reps ({role})",
                ):
                    tok, rep = fut.result()
                    if tok not in sums:
                        sums[tok] = rep.copy()
                        counts[tok] = 1
                    else:
                        sums[tok] += rep
                        counts[tok] += 1

        extension = {tok: (sums[tok] / counts[tok]) for tok in sums.keys()}
        extension = _post_normalize(extension)

        for tok, vec in extension.items():
            self.extensions[role][tok] = np.asarray(vec)
            self.extended_tokens[role].add(tok)
            self.extension_counts[role][tok] = int(counts[tok])

        self.extension_lengths[role] = len(self.extensions[role])
        self._sync_extension_flags()

        return extension

    def select_top_k(self, role: str, top_k: int):
        self._validate_role(role)

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")

        n_ext = len(self.extensions[role])
        if n_ext < top_k:
            raise ValueError(
                f"Not enough extended tokens for role {role!r}: "
                f"have {n_ext}, requested top_k={top_k}"
            )

        counts = self.extension_counts[role]
        ranked = sorted(self.extensions[role].keys(), key=lambda t: (-counts.get(t, 0), t))
        keep = set(ranked[:top_k])

        drop = [tok for tok in self.extensions[role].keys() if tok not in keep]
        for tok in drop:
            self.extensions[role].pop(tok, None)
            self.extended_tokens[role].discard(tok)
            self.extension_counts[role].pop(tok, None)

        self.extension_lengths[role] = len(self.extensions[role])
        self._sync_extension_flags()

        return ranked[:top_k]

    def integrate_extension(self, top_k: int | None = None) -> TuckerDecomposition:
        """
        Materialize extension vectors into the factor matrices + vocab,
        returning a plain TuckerDecomposition.
        """
        # Preserve linked/shared factors if configured
        if getattr(self, "shared_factors", None):
            for a, b in self.shared_factors:
                role_a, role_b = self.roles[a], self.roles[b]

                combined_toks = (
                    set(self.extensions[role_a].keys()) |
                    set(self.extensions[role_b].keys())
                )

                for tok in combined_toks:
                    vecs = []
                    counts = 0

                    if tok in self.extensions[role_a]:
                        c = self.extension_counts[role_a][tok]
                        vecs.append(self.extensions[role_a][tok] * c)
                        counts += c

                    if tok in self.extensions[role_b]:
                        c = self.extension_counts[role_b][tok]
                        vecs.append(self.extensions[role_b][tok] * c)
                        counts += c

                    avg_vec = sum(vecs) / counts

                    self.extensions[role_a][tok] = avg_vec
                    self.extensions[role_b][tok] = avg_vec
                    self.extended_tokens[role_a].add(tok)
                    self.extended_tokens[role_b].add(tok)
                    self.extension_counts[role_a][tok] = counts
                    self.extension_counts[role_b][tok] = counts

                self.extension_lengths[role_a] = len(self.extensions[role_a])
                self.extension_lengths[role_b] = len(self.extensions[role_b])

        top_ks: dict[str, int] = {}
        if top_k is not None:
            for role in self.roles:
                n_ext = self.extension_lengths[role]
                if n_ext < top_k:
                    raise ValueError(
                        f"Not enough extended tokens for role {role!r}: "
                        f"have {n_ext}, requested top_k={top_k}"
                    )
                if n_ext > top_k:
                    self.select_top_k(role, top_k)
                top_ks[role] = top_k
        else:
            top_ks = {role: self.extension_lengths[role] for role in self.roles}

        new_vocab = dict(self.vocab)
        for role in self.roles:
            list_key = _voc_list_key(role)
            map_key = voc_index(role)

            new_vocab[list_key] = list(new_vocab[list_key])
            new_vocab[map_key] = dict(new_vocab[map_key])

        new_factors: List[Union[torch.Tensor, np.ndarray]] = []
        for role in self.roles:
            f_idx = self.get_role_index(role)
            F = self.factors[f_idx]
            counts = self.extension_counts[role]

            if not counts:
                new_factors.append(F)
                continue

            toks = sorted(self.extensions[role].keys(), key=lambda t: (-counts.get(t, 0), t))
            toks = toks[:top_ks[role]]

            vecs_np = np.stack([np.asarray(self.extensions[role][tok]) for tok in toks], axis=0)

            if isinstance(F, torch.Tensor):
                add = torch.tensor(vecs_np, dtype=F.dtype, device=F.device)
                F_new = torch.cat([F, add], dim=0)
            else:
                F_np = _to_np(F)
                F_new = np.vstack([F_np, vecs_np])

            new_factors.append(F_new)

            list_key = _voc_list_key(role)
            map_key = voc_index(role)

            base_n = len(new_vocab[list_key])
            for j, tok in enumerate(toks):
                new_vocab[list_key].append(tok)
                new_vocab[map_key][tok] = base_n + j

        return TuckerDecomposition(
            self.core,
            new_factors,
            new_vocab,
            shared_factors=self.shared_factors,
            roles=self.roles,
        )

    def save_extensions(
        self,
        path: str,
        *,
        roles: Optional[list[str]] = None,
    ) -> None:
        """
        Save ONLY extension vectors and metadata needed to restore them.
        """
        if roles is None:
            roles = list(self.roles)

        for role in roles:
            self._validate_role(role)

        R = self.factors[0].shape[1]

        payload = {
            "rank": R,
            "roles_order": list(self.roles),
            "is_extended": bool(self.is_extended),
            "extended_roles": sorted(list(self.extended_roles)),
            "extension_lengths": dict(self.extension_lengths),
            "roles": {},
        }

        for role in roles:
            counts = self.extension_counts.get(role, {})
            toks = sorted(self.extensions[role].keys(), key=lambda t: (-counts.get(t, 0), t))

            if len(toks) == 0:
                payload["roles"][role] = {
                    "tokens": [],
                    "counts": [],
                    "matrix": None,
                    "dtype": None,
                }
                continue

            mat = np.stack([np.asarray(self.extensions[role][tok]) for tok in toks], axis=0)
            if mat.shape[1] != R:
                raise ValueError(
                    f"Extension rank mismatch for role {role}: got {mat.shape[1]}, expected {R}"
                )

            role_counts = [int(counts.get(tok, 0)) for tok in toks]

            payload["roles"][role] = {
                "tokens": toks,
                "counts": role_counts,
                "matrix": torch.from_numpy(mat),
                "dtype": str(mat.dtype),
            }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(payload, path)

    def load_extensions_inplace(
        self,
        path: str,
        *,
        map_location: Union[str, torch.device] = "cpu",
        strict_rank: bool = True,
        overwrite: bool = False,
    ) -> None:
        """
        Load saved extensions into this ExtendedTucker instance.
        """
        try:
            payload = torch.load(path, map_location=map_location)
        except Exception:
            import pickle
            with open(path, "rb") as f:
                payload = pickle.load(f)

        saved_R = int(payload["rank"])
        cur_R = self.factors[0].shape[1]
        if strict_rank and saved_R != cur_R:
            raise ValueError(f"Rank mismatch: file rank={saved_R}, current rank={cur_R}")

        roles_blob = payload.get("roles", {})
        for role, blob in roles_blob.items():
            self._validate_role(role)

            toks = blob.get("tokens", []) or []
            counts = blob.get("counts", []) or []
            mat = blob.get("matrix", None)

            if overwrite:
                self.extensions[role].clear()
                self.extended_tokens[role].clear()
                self.extension_counts[role].clear()

            if mat is None or len(toks) == 0:
                self.extension_lengths[role] = len(self.extensions[role])
                continue

            if isinstance(mat, torch.Tensor):
                mat_np = mat.detach().cpu().numpy()
            else:
                mat_np = np.asarray(mat)

            if mat_np.ndim != 2 or mat_np.shape[1] != cur_R:
                raise ValueError(
                    f"Bad matrix shape in file for role {role!r}: {mat_np.shape}, expected (n,{cur_R})"
                )
            if len(counts) != len(toks):
                raise ValueError(f"Counts length != tokens length for role {role!r}")

            for i, tok in enumerate(toks):
                if (not overwrite) and (tok in self.extensions[role]):
                    continue
                vec = np.asarray(mat_np[i], dtype=np.float64)
                self.extensions[role][tok] = vec
                self.extended_tokens[role].add(tok)
                self.extension_counts[role][tok] = int(counts[i])

            self.extension_lengths[role] = len(self.extensions[role])

        self._sync_extension_flags()

    @classmethod
    def load_extensions(
        cls,
        t: TuckerDecomposition,
        path: str,
        *,
        map_location: Union[str, torch.device] = "cpu",
        strict_rank: bool = True,
        overwrite: bool = False,
    ) -> "ExtendedTucker":
        ext = cls.from_tucker(t)
        ext.load_extensions_inplace(
            path,
            map_location=map_location,
            strict_rank=strict_rank,
            overwrite=overwrite,
        )
        return ext
