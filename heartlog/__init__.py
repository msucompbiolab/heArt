"""Canonical Advanced/Raw Log standard for heArt_legacy.

One shared, levelled, rank-aware logger for every solver, workflow and process in this
repository. The record layout is shared verbatim with the service node, which renders this
repository's stdout as the Raw log inside its ``Advanced`` panel::

    HH:MM:SS.mmm  LEVEL  source/stage   message  key=value key=value

Usage::

    from heartlog import get_logger
    log = get_logger("unload")

    log.info("anchor", "Klotz V0 anchor resolved", v0_ml=97.4, cparam_pa=210.0)
    log.warn("mff", "fit exhausted; stopping the FIT, not the unload", it=4)

    with log.stage("unloading", edp_mmhg=12.0) as st:
        for i, p in enumerate(schedule):
            ...
            st.progress(i + 1, len(schedule), p_mmhg=p, v_ml=vol)
        st.outputs(v0_ml=v0)

which yields one ``BEGIN`` record, dual-throttled ``progress`` records, and exactly one
terminal ``END``/``FAIL`` record carrying ``elapsed=``.

This package is stdlib-only and MUST stay free of ``dolfin`` and ``mpi4py`` imports so the
dolfin-free control-plane packages (``autocore``, ``autocal``, ``active_lab``, ...) can use it.
The DOLFIN-aware half -- MPI rank resolution and the single DOLFIN/PETSc log-configuration
point -- lives in ``src/utils/log_mpi.py``.
"""
from __future__ import annotations

from .format import LEVELS, format_kv, format_line, infer_level
from .logger import (
    Logger,
    StageCtx,
    all_ranks_enabled,
    announce_once,
    get_logger,
    min_level,
    reset_state,
)
from .stages import STAGES, is_known

__all__ = [
    "LEVELS",
    "Logger",
    "STAGES",
    "StageCtx",
    "all_ranks_enabled",
    "announce_once",
    "format_kv",
    "format_line",
    "get_logger",
    "infer_level",
    "is_known",
    "min_level",
    "reset_state",
]
