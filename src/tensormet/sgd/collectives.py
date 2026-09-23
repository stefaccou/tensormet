"""
collectives.py — cross-device sum-reductions for the
single-process SGD trainers.

A ``Collective`` sums G tensors (one per device) in place, leaving the result
on every device. It is also the seam a DDP backend would slot into (see
sgd/README.md).

Backends
--------
``NcclSingleProcess``  ``torch.cuda.nccl.all_reduce`` (single-process,
                       multi-device). Preferred.
``HostReduce``         Pinned host staging; fallback when NCCL or peer access
                       is missing. Costs ``(G-1) x buffer`` pinned bytes.
``SingleDevice``       No-op, so both trainers share one code path.

Buffers are cached by ``(numel, dtype)``, so nothing is allocated in steady
state.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Dict, List, Sequence, Tuple

import torch

#: Seconds to wait for the construction-time NCCL probe before giving up on it.
#: Comm init is milliseconds when it works, so this is a hang detector, not a
#: budget; override for a pathologically slow node.
NCCL_PROBE_TIMEOUT_S = float(os.environ.get("TENSORMET_NCCL_PROBE_TIMEOUT", "30"))


class Collective:
    """Sum-reduce one tensor per device, in place, result on every device.

    Subclasses implement ``all_reduce``; ``all_reduce_scalar`` and
    ``broadcast_`` have generic implementations that only the no-op backend
    bothers to override.
    """

    #: devices this collective spans, in order (index == replica index)
    devices: List[torch.device]

    @property
    def n_devices(self) -> int:
        return len(self.devices)

    def all_reduce(self, tensors: Sequence[torch.Tensor]) -> None:
        """``tensors[g]`` lives on ``devices[g]``; all become the sum."""
        raise NotImplementedError

    def all_reduce_scalar(self, tensors: Sequence[torch.Tensor]) -> None:
        """Same contract for tiny (0-d / 1-element) tensors.

        Split out because Phase 4 (core sharding) reduces a scalar per step
        while the gradient path reduces hundreds of megabytes, and a backend
        may reasonably want different staging for the two.
        """
        self.all_reduce(tensors)

    def broadcast_(self, tensors: Sequence[torch.Tensor]) -> None:
        """``tensors[0]`` -> every other device. Only needed off the hot path
        (resume, and the periodic parameter averaging of ``sgd_sync_every``)."""
        src = tensors[0]
        for dst in tensors[1:]:
            dst.copy_(src, non_blocking=True)

    def sync(self) -> None:
        """Block until every device is idle. Used around timing and checkpoints."""
        for dev in self.devices:
            torch.cuda.synchronize(dev)


class SingleDevice(Collective):
    """G == 1: every collective is a no-op."""

    def __init__(self, device: torch.device):
        self.devices = [torch.device(device)]

    def all_reduce(self, tensors: Sequence[torch.Tensor]) -> None:
        return

    def broadcast_(self, tensors: Sequence[torch.Tensor]) -> None:
        return

    def sync(self) -> None:
        if self.devices[0].type == "cuda":
            torch.cuda.synchronize(self.devices[0])

    def __repr__(self) -> str:
        return f"SingleDevice({self.devices[0]})"


class NcclSingleProcess(Collective):
    """``torch.cuda.nccl.all_reduce`` over a list of one tensor per device.

    This is NCCL's single-process multi-device mode: the caller hands it G
    tensors that live on G different devices and it performs a ring all-reduce
    between them. No process group, no rank assignment, no ``torchrun``.
    """

    def __init__(self, devices: Sequence[torch.device]):
        self.devices = [torch.device(d) for d in devices]

    def all_reduce(self, tensors: Sequence[torch.Tensor]) -> None:
        # nccl.all_reduce requires a *list*, contiguous tensors, one per device.
        torch.cuda.nccl.all_reduce(list(tensors))

    def __repr__(self) -> str:
        return f"NcclSingleProcess({[d.index for d in self.devices]})"


class HostReduce(Collective):
    """Pinned-host staging + device-0 accumulation + scatter back.

    Staging buffers are allocated once per ``(numel, dtype)`` and reused; the
    accumulation is an in-place ``add_``.

    The host hop is deliberate: this backend exists precisely for topologies
    where devices cannot reach each other directly, and pinned memory makes the
    two hops asynchronous-capable rather than pageable-slow.
    """

    def __init__(self, devices: Sequence[torch.device]):
        self.devices = [torch.device(d) for d in devices]
        # (numel, dtype) -> (host staging per source device, device-0 scratch
        # per source device). Separate scratch per source so the H2D copies can
        # overlap instead of serializing on one buffer.
        self._cache: Dict[Tuple[int, torch.dtype], Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}

    def _buffers(self, numel: int, dtype: torch.dtype):
        key = (int(numel), dtype)
        buf = self._cache.get(key)
        if buf is None:
            n_src = len(self.devices) - 1
            host = [torch.empty(numel, dtype=dtype, pin_memory=True) for _ in range(n_src)]
            scratch = [torch.empty(numel, dtype=dtype, device=self.devices[0])
                       for _ in range(n_src)]
            buf = (host, scratch)
            self._cache[key] = buf
        return buf

    def all_reduce(self, tensors: Sequence[torch.Tensor]) -> None:
        if len(tensors) < 2:
            return
        flat = [t.reshape(-1) for t in tensors]
        host, scratch = self._buffers(flat[0].numel(), flat[0].dtype)
        # D2H for every replica first, so the G-1 device streams drain in
        # parallel rather than one-at-a-time behind the accumulation.
        for i, t in enumerate(flat[1:]):
            host[i].copy_(t)
        acc = flat[0]
        for i in range(len(flat) - 1):
            scratch[i].copy_(host[i], non_blocking=True)
            acc.add_(scratch[i])
        # Scatter the sum back so every device sees the same value, matching
        # the all-reduce contract the NCCL backend provides.
        for i, t in enumerate(flat[1:]):
            host[i].copy_(acc)
            t.copy_(host[i], non_blocking=True)

    def __repr__(self) -> str:
        return f"HostReduce({[d.index for d in self.devices]})"


def _peer_access_ok(devices: Sequence[torch.device]) -> bool:
    """True when every ordered pair of distinct devices can reach the other."""
    idx = [d.index for d in devices]
    if any(i is None for i in idx):
        return False
    for a in idx:
        for b in idx:
            if a != b and not torch.cuda.can_device_access_peer(a, b):
                return False
    return True


def _nccl_available(devices: Sequence[torch.device]) -> bool:
    """NCCL is Linux-only and optional in some builds, and probing it needs a
    live tensor per device — so ask once, defensively, at construction."""
    try:
        return bool(torch.cuda.nccl.is_available(
            [torch.empty(1, device=d) for d in devices]
        ))
    except (RuntimeError, AttributeError, AssertionError):
        return False


class _ProbeTimeout(Exception):
    """The NCCL probe did not return within ``NCCL_PROBE_TIMEOUT_S``."""


def _call_with_timeout(fn: Callable[[], bool], timeout: float) -> bool:
    """Run ``fn()`` on a daemon thread and raise ``_ProbeTimeout`` if it stalls.

    A stalled C call can't be cancelled, so the thread is abandoned as a
    daemon: a small one-off leak, preferable to the job hanging until walltime.
    """
    box: Dict[str, object] = {}

    def _target() -> None:
        try:
            box["ok"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
            box["exc"] = exc

    thread = threading.Thread(target=_target, daemon=True, name="nccl-probe")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise _ProbeTimeout(f"no result after {timeout:g}s")
    exc = box.get("exc")
    if exc is not None:
        raise exc  # type: ignore[misc]
    return bool(box.get("ok", False))


def _nccl_works(devices: Sequence[torch.device]) -> bool:
    """Actually run a 1-element all-reduce and check the answer.

    ``is_available`` never initializes a communicator, and comm init is where
    NCCL fails on clusters (P2P/IPC limits, small ``/dev/shm``, MIG, Hopper
    NVLS) — otherwise only on the first real reduction. Comm init usually
    *blocks* rather than raises, so the probe runs under a watchdog and a
    timeout counts as failure.
    """
    def _probe() -> bool:
        probe = [torch.full((1,), 1.0, device=d) for d in devices]
        torch.cuda.nccl.all_reduce(probe)
        for d in devices:
            torch.cuda.synchronize(d)
        expected = float(len(devices))
        return all(float(p.item()) == expected for p in probe)

    try:
        return _call_with_timeout(_probe, NCCL_PROBE_TIMEOUT_S)
    except _ProbeTimeout as exc:
        print(
            f"WARNING: NCCL probe over {[d.index for d in devices]} hung ({exc}); "
            "treating NCCL as unusable on this node. Common causes: Hopper "
            "NVLS/multicast (NCCL_NVLS_ENABLE=0), restricted P2P/IPC "
            "(NCCL_P2P_DISABLE=1), or a small /dev/shm under the job scheduler "
            "(NCCL_SHM_DISABLE=1). Re-run with NCCL_DEBUG=INFO for the reason.",
            flush=True,
        )
        return False
    except (RuntimeError, AttributeError, AssertionError):
        return False


def make_collective(
    devices: Sequence[torch.device],
    backend: str = "auto",
) -> Collective:
    """Build the collective for ``devices``.

    ``backend``:
      ``"auto"``  NCCL when it is available, the devices have a full
                  peer-access mesh, *and* a probe all-reduce actually succeeds;
                  ``HostReduce`` otherwise; ``SingleDevice`` for G == 1.
      ``"nccl"``  force NCCL (raises if unavailable or if the probe fails — a
                  silent fall back to the slow path would quietly explain away
                  a bad benchmark).
      ``"host"``  force the pinned-staging path.
    """
    devs = [torch.device(d) for d in devices]
    if backend not in ("auto", "nccl", "host"):
        raise ValueError(
            f"sgd_comm_backend must be 'auto', 'nccl' or 'host'; got {backend!r}"
        )
    if len(devs) == 1:
        return SingleDevice(devs[0])
    if any(d.type != "cuda" for d in devs):
        raise ValueError(f"multi-device collectives need CUDA devices; got {devs}")

    if backend == "host":
        return HostReduce(devs)

    nccl_ok = _nccl_available(devs)
    if backend == "nccl":
        if not nccl_ok:
            raise RuntimeError(
                "sgd_comm_backend='nccl' requested but torch.cuda.nccl is not "
                "available for these devices; use 'host' or 'auto'."
            )
        if not _nccl_works(devs):
            raise RuntimeError(
                f"sgd_comm_backend='nccl' requested but a probe all-reduce over "
                f"{[d.index for d in devs]} failed or hung. Re-run with "
                "NCCL_DEBUG=INFO for the reason; common ones are Hopper "
                "NVLS/multicast (try NCCL_NVLS_ENABLE=0), restricted P2P/IPC "
                "between these GPUs (NCCL_P2P_DISABLE=1), a small /dev/shm under "
                "the job scheduler (NCCL_SHM_DISABLE=1), or MIG slices "
                "(single-process multi-device NCCL cannot span them). "
                "sgd_comm_backend='host' always works."
            )
        return NcclSingleProcess(devs)

    # auto: peer access is the cheap pre-filter, the probe is the real test.
    if nccl_ok and _peer_access_ok(devs) and _nccl_works(devs):
        return NcclSingleProcess(devs)
    return HostReduce(devs)


__all__ = [
    "Collective",
    "SingleDevice",
    "NcclSingleProcess",
    "HostReduce",
    "make_collective",
]
