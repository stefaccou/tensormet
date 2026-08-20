"""Submitit-based dispatch for tensormet Tier-2 (Slurm) jobs.

Motivation
----------
The hand-written ``#SBATCH`` job scripts under ``scripts/9_hpc`` have one
structural weakness: Slurm does **not** expand shell/environment variables in
``#SBATCH`` directives, and a relative ``--output=logs/...`` is resolved against
the *submission* working directory. So ``$PROJ/metaphor/logs`` is unreachable
from a directive, and logs leak into ``scripts/9_hpc/logs`` depending on where
``sbatch`` happened to be run.

Submitit sidesteps this entirely: the log location is an ordinary Python value
(the executor ``folder``), so it honours ``$PROJ`` via ``os.path.expandvars``.
The scheduler header is built from one :class:`SlurmProfile` dataclass instead
of duplicated directive blocks, and preemption / walltime timeouts requeue
automatically through :class:`DecompJob.checkpoint`, piggy-backing on the
existing ``--resume`` + node-local staging machinery (see ``hpc_helpers``).

Usage
-----
Run from the login node **with the tensormet venv active** (submitit reuses
``sys.executable`` on the compute node, so the interpreter you submit from is
the one that runs the job)::

    venv
    from tensormet.experimental.submit import DecompJob, build_executor, TIER2_A100_SMOKE
    ex  = build_executor("$PROJ/metaphor/logs", TIER2_A100_SMOKE, job_name="smoke")
    job = ex.submit(DecompJob(["--dataset", "...", "--dim", "1000", ...]))
    print(job.job_id)

The argv list is exactly what the old scripts passed to
``python -m tensormet.scripts.nnt``, so behaviour is identical.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import submitit

from tensormet.parsing import parse_run_config
from tensormet.launch import launch_nnt_decomposition


# --------------------------------------------------------------------------- #
# Job payload
# --------------------------------------------------------------------------- #
def _with_resume(argv: List[str], value: str = "t") -> List[str]:
    """Return a copy of ``argv`` with ``--resume`` forced to ``value``.

    Used on requeue: a preempted/timed-out job must continue from its last
    checkpoint rather than restart, regardless of the ``--resume`` it was first
    submitted with. Strips any existing ``--resume X`` pair, then appends one.
    """
    out: List[str] = []
    skip = False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok == "--resume":
            skip = True  # also drop the following value
            continue
        out.append(tok)
    out += ["--resume", value]
    return out


class DecompJob(submitit.helpers.Checkpointable):
    """Picklable, resubmittable decomposition job.

    Holds the CLI argv the hand-written scripts fed to
    ``python -m tensormet.scripts.nnt`` and, when called on the compute node,
    rebuilds the exact same :class:`RunConfig` and launches it. Because it
    subclasses :class:`submitit.helpers.Checkpointable`, submitit invokes
    :meth:`checkpoint` when the job is preempted or hits its walltime; we
    resubmit the same argv with ``--resume t`` so training picks up from the
    last checkpoint the staging logic mirrored back to durable storage.
    """

    def __init__(self, argv: List[str]):
        self.argv: List[str] = list(argv)

    def __call__(self):
        cfg = parse_run_config(self.argv)
        print(f"[submit] run_id={cfg.run_id()} argv={self.argv}", flush=True)
        return launch_nnt_decomposition(cfg)

    def checkpoint(self, *args, **kwargs) -> submitit.helpers.DelayedSubmission:
        # Requeue ourselves with resume forced on; submitit reschedules this.
        return submitit.helpers.DelayedSubmission(DecompJob(_with_resume(self.argv)))


# --------------------------------------------------------------------------- #
# Scheduler profiles
# --------------------------------------------------------------------------- #
@dataclass
class SlurmProfile:
    """One Tier-2 (VSC wice) scheduler header, expressed as data.

    Field names mirror the ``#SBATCH`` directives they replace so the mapping
    to the old scripts is obvious. ``setup`` are shell lines injected into the
    batch script before the job launches (module loads / venv activation),
    matching the preamble the old scripts ran by hand.
    """

    partition: str
    gpus_per_node: int
    cpus_per_task: int
    mem_gb: int
    timeout_min: int
    account: str = "lp_tenacity"
    clusters: str = "wice"
    nodes: int = 1
    tasks_per_node: int = 1
    array_parallelism: Optional[int] = None
    setup: List[str] = field(
        default_factory=lambda: [
            "source $VSC_HOME/.bashrc",
            "venv",
            "export PYTHONUNBUFFERED=1",
        ]
    )


# Tiny single-GPU smoke-test profiles (tier-2 / wice), one per GPU type.
# Match the resource shape of 2026-07-01-tier2-h100_smoketest.sh: 1 GPU,
# ~one GPU's core+memory share, 30 min walltime.
TIER2_H100_SMOKE = SlurmProfile(
    partition="gpu_h100",
    gpus_per_node=1,
    cpus_per_task=16,   # wice H100 node: 64 cores / 4 GPUs
    mem_gb=120,
    timeout_min=30,
)

TIER2_A100_SMOKE = SlurmProfile(
    partition="gpu_a100",
    gpus_per_node=1,
    cpus_per_task=18,   # wice A100 node: 72 cores / 4 GPUs
    mem_gb=120,
    timeout_min=30,
)

# Two-GPU profile for the sharded paths (MU ShardedSparseTensor with
# --n_gpus 2, or the SGD ShardedSGDTrainer). Still one node / one task:
# both shard within a single process, so tasks_per_node stays 1.
TIER2_H100_DUAL = SlurmProfile(
    partition="gpu_h100",
    gpus_per_node=2,
    cpus_per_task=32,   # two GPUs' core share on a 64-core / 4-GPU node
    mem_gb=240,
    timeout_min=120,
)

# Whole-node profile: the SGD trainer's thread-per-GPU fan-out and single
# all-reduce per step are sized for up to 4 devices in ONE process, so
# tasks_per_node stays 1 here too — there is no torchrun/srun rank layout to
# arrange. The CPU share matters more than for MU: each GPU gets a dispatch
# thread, and a dispatch-bound step is exactly the regime where a starved host
# thread shows up as lost scaling.
TIER2_H100_QUAD = SlurmProfile(
    partition="gpu_h100",
    gpus_per_node=4,
    cpus_per_task=64,   # the whole 64-core / 4-GPU wice H100 node
    mem_gb=480,
    timeout_min=120,
)


# --------------------------------------------------------------------------- #
# Executor construction
# --------------------------------------------------------------------------- #
def _expand(p) -> Path:
    """Expand ``$VARS`` / ``~`` in a path *in Python* (the thing SBATCH can't)."""
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


def build_executor(
    log_dir,
    profile: SlurmProfile,
    job_name: str,
) -> "submitit.AutoExecutor":
    """Build a submitit executor that logs under ``log_dir``.

    ``log_dir`` may contain ``$PROJ`` (or any env var); it is expanded here, so
    passing ``"$PROJ/metaphor/logs"`` puts ``<jobid>_*_log.out/.err`` exactly
    there. The directory is created if missing (Slurm itself never creates log
    dirs — a missing one silently drops output).
    """
    folder = _expand(log_dir)
    folder.mkdir(parents=True, exist_ok=True)

    ex = submitit.AutoExecutor(folder=str(folder))
    ex.update_parameters(
        name=job_name,
        nodes=profile.nodes,
        tasks_per_node=profile.tasks_per_node,
        cpus_per_task=profile.cpus_per_task,
        gpus_per_node=profile.gpus_per_node,
        mem_gb=profile.mem_gb,
        timeout_min=profile.timeout_min,
        slurm_account=profile.account,
        # VSC wice needs an explicit cluster; no generic submitit param for it.
        slurm_additional_parameters={"clusters": profile.clusters},
        slurm_setup=profile.setup,
    )
    if profile.array_parallelism is not None:
        ex.update_parameters(slurm_array_parallelism=profile.array_parallelism)
    return ex
