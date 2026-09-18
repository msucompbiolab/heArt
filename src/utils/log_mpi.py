"""DOLFIN-aware half of the logging standard: MPI rank resolution and the single
DOLFIN/PETSc log-configuration point.

``heartlog`` itself is stdlib-only so the dolfin-free control-plane packages can import it.
Everything that needs ``dolfin`` lives here.

Deliberately NOT re-exported from ``src/utils/__init__.py``: that module wildcard-imports the
whole FE stack, so re-exporting would drag DOLFIN into any package that only wanted a logger.

**Collective-free.** Rank resolution reads a communicator's own rank and nothing else -- no
gather, no barrier, no reduction. A collective on the logging path deadlocks the moment two
ranks take different branches, which is exactly what a warning-on-one-rank is.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import dolfin as df

# Both import spellings resolve when the repo AND its parent are on sys.path (what the demo
# scripts and every package bootstrap set up). A caller with only ONE of them still works: this
# is the single place in the FE stack that reaches for heartlog, and `infer_level` is re-exported
# so no other src/ module has to repeat the fallback.
try:
    from heartlog import Logger, announce_once, infer_level
except ImportError:  # pragma: no cover - only when the repo dir itself is not importable
    from heArt_legacy.heartlog import Logger, announce_once, infer_level

__all__ = [
    "Logger", "announce_once", "infer_level", "rank_of", "is_root", "rank_fn_for",
    "get_logger", "configure_dolfin_logging", "configure_petsc_monitors",
]


def rank_of(comm: Any = None) -> int:
    """Rank of ``comm``, tolerating a DOLFIN comm, an mpi4py comm, or ``None``.

    Generalises the one ad-hoc fallback that previously existed (``has_dataset_parallel``), so
    every call site gets the same tolerance instead of each guessing at the comm flavour.
    """
    if comm is None:
        comm = df.MPI.comm_world
    try:
        return int(df.MPI.rank(comm))
    except Exception:
        pass
    try:
        return int(comm.rank)
    except Exception:
        pass
    try:
        return int(comm.Get_rank())
    except Exception:
        pass
    # Unknown comm flavour: assume root so narration is never lost, and say so once.
    if announce_once("log_mpi:unknown_comm"):
        import sys
        sys.stderr.write(
            "WARNING [log_mpi]: unrecognised communicator %r; assuming rank 0 -- under MPI its "
            "records may be duplicated once per rank.\n" % (type(comm).__name__,)
        )
        sys.stderr.flush()
    return 0


def is_root(comm: Any = None) -> bool:
    """Single replacement for the five rank-guard idioms and six duplicated ``_rank0()``."""
    return rank_of(comm) == 0


def rank_fn_for(comm: Any = None) -> Callable[[], int]:
    """A late-binding rank resolver for a :class:`~heartlog.Logger`.

    Late binding matters: a logger is often constructed at module import, before the mesh (and
    therefore the communicator) exists.
    """
    return lambda: rank_of(comm)


def get_logger(source: str, comm: Any = None) -> Logger:
    """A logger whose rank gate follows ``comm`` (defaults to ``comm_world``)."""
    return Logger(source, rank_fn=rank_fn_for(comm))


def configure_dolfin_logging(comm: Any = None, *, level: Optional[Any] = None) -> None:
    """The single configuration point for DOLFIN's own log stream.

    Replaces the scattered ``df.set_log_level(...)`` calls at the top of each entry point and
    the per-solve global side effect that used to live inside ``NSolver.solvenonlinear``.

    Preserves the deliberate anti-flood contract: DOLFIN's own output is confined to rank 0 and
    raised to WARNING, which suppresses the ~5 lines PER Newton solve ("Solving nonlinear
    variational problem." plus the full per-iteration report) that otherwise bury real stage
    progress on a run with hundreds of solves. Genuine solver warnings and errors still pass,
    and non-convergence still RAISES, so nothing is masked.
    """
    df.set_log_active(is_root(comm))
    df.set_log_level(df.LogLevel.WARNING if level is None else level)


def configure_petsc_monitors(enabled: bool, log: Optional[Logger] = None) -> None:
    """Enable PETSc's SNES/KSP monitors on demand.

    These were previously hardcoded on for the SNES path and rank-unguarded, so every rank wrote
    per-iteration residuals straight to ``PETSC_COMM_WORLD``'s stdout -- thousands of unprefixed
    lines that no consumer reads and that drown the Raw log. They are now opt-in
    (``SimDet["petsc_monitors"]``) and announce themselves when engaged, so the noise is never a
    surprise.
    """
    from petsc4py import PETSc

    opts = PETSc.Options()
    keys = ("snes_monitor", "ksp_monitor_true_residual", "snes_converged_reason")
    if not enabled:
        for key in keys:
            try:
                opts.delValue(key)
            except Exception:
                pass
        return
    for key in keys:
        opts.setValue(key, "")
    if log is not None and announce_once("log_mpi:petsc_monitors"):
        log.warn(
            "init",
            "PETSc SNES/KSP monitors ENABLED: per-iteration residuals are written unprefixed "
            "by every rank and will interleave with the structured records",
            options=",".join(keys),
        )
