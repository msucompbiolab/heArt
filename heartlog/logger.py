"""The shared logger: levelled, rank-aware, stage-bracketed, collective-free.

Design constraints specific to this repository:

* **Collective-free.** Nothing here performs an MPI gather, barrier or reduction. A logging
  path that synchronises ranks can deadlock the moment two ranks take different branches (a
  failure mode this codebase has already hit) and would throttle the solve. Rank 0 emits by
  default; ``HEART_LOG_ALL_RANKS=1`` makes every rank emit with a ``rank=`` field. Ordering is
  therefore guaranteed per rank -- cross-rank interleave is whatever ``mpirun`` delivers.

* **Import-identity safe.** This package is reachable as both ``heartlog`` (control-plane
  packages import bare) and ``heArt_legacy.heartlog`` (the FE side imports through the parent
  directory). Those are two distinct module objects with independent globals, so any state that
  must be process-wide -- progress throttles, "announce once" guards -- lives on a single
  sentinel module registered under a fixed ``sys.modules`` key, never in module globals.

* **Best-effort but never silent.** An emission fault must not kill a running solve, so writes
  are guarded. Per the repository's error-handling principle a degraded path may never be
  silent: the first failure emits a one-shot WARNING to stderr naming the cause and the
  degraded behaviour now in effect.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Callable, Iterator, Optional

from .format import default_clock, format_line, level_rank
from .stages import is_known

# --- process-wide state (see the module docstring on import identity) --------------------
_STATE_KEY = "_heartlog_state"


def _state() -> ModuleType:
    st = sys.modules.get(_STATE_KEY)
    if st is None:
        st = ModuleType(_STATE_KEY)
        st.announced = set()       # one-shot guard keys
        st.emit_failed = False     # the logger's own degraded-path flag
        sys.modules[_STATE_KEY] = st
    return st


def announce_once(key: str) -> bool:
    """True the FIRST time ``key`` is seen in this process; False afterwards.

    The shared replacement for the per-class one-shot flags that were duplicated across
    solvers (e.g. the MUMPS null-pivot warning), so a per-solve warning cannot flood a log.
    """
    st = _state()
    if key in st.announced:
        return False
    st.announced.add(key)
    return True


def reset_state() -> None:
    """Clear process-wide logger state. Tests only."""
    sys.modules.pop(_STATE_KEY, None)


# --- environment knobs -------------------------------------------------------------------
def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def all_ranks_enabled() -> bool:
    """Whether every rank emits (default: rank 0 only)."""
    return _env_flag("HEART_LOG_ALL_RANKS", False)


def min_level() -> str:
    """Lowest level that is emitted; anything below is dropped before formatting."""
    return os.environ.get("HEART_LOG_LEVEL", "INFO").strip().upper()


def progress_min_interval_s() -> float:
    return _env_float("HEART_LOG_PROGRESS_MIN_INTERVAL_S", 1.0)


def progress_min_pct() -> float:
    return _env_float("HEART_LOG_PROGRESS_MIN_PCT", 5.0)


class StageCtx:
    """Handle yielded by :meth:`Logger.stage` for progress ticks and END-line outputs."""

    def __init__(self, log: "Logger", stage: str) -> None:
        self._log = log
        self._stage = stage
        self._outputs: dict[str, Any] = {}
        # Seeded at entry so the very first tick is throttled like any other. Advance is measured
        # from the last EMITTED pct, so a suppressed tick does not reset the baseline.
        self._last_emit_at = log._clock()
        self._last_emit_pct = 0.0
        self._max_pct = 0.0

    def outputs(self, **kv: Any) -> None:
        """Record values to fold into this stage's END record."""
        self._outputs.update(kv)

    def due(self) -> bool:
        """Cheap pre-check: would a :meth:`progress` call be able to emit right now?

        Guard any EXPENSIVE detail with this. Arguments are evaluated before the call, so
        ``st.progress(i, n, v_lv_ml=model.get_lv_volume())`` would run an FE assembly on every
        timestep only for the record to be thrown away by the throttle. This tests the time
        guard alone (one clock read) -- ``progress`` still re-checks both guards, so a caller
        that skips this is correct, just wasteful.
        """
        return (self._log._clock() - self._last_emit_at) >= progress_min_interval_s()

    def progress(self, done: float, total: float, /, **kv: Any) -> None:
        """Report ``done``/``total`` within this stage. Dual-throttled and monotone.

        Emits only once BOTH guards clear (min elapsed AND min percent advance), so a tight
        per-timestep loop stays quiet while a slow single solve still ticks about once a second.
        On the throttled path this costs one clock read and two comparisons -- the record is
        never formatted when the throttle is closed.
        """
        if total <= 0:
            return
        local_pct = max(0.0, min(100.0, (float(done) / float(total)) * 100.0))
        # Never walk backwards: a coupling backoff rewinds `done`, and a bar that goes backwards
        # reads as a fault to someone watching the stream.
        local_pct = max(self._max_pct, local_pct)
        self._max_pct = local_pct

        now = self._log._clock()
        if (now - self._last_emit_at) < progress_min_interval_s():
            return
        if (local_pct - self._last_emit_pct) < progress_min_pct() and local_pct < 100.0:
            return
        self._last_emit_at = now
        self._last_emit_pct = local_pct
        # Merge rather than splat both, so a caller's own `done=`/`total=` detail cannot collide.
        detail: dict[str, Any] = {"done": done, "total": total}
        detail.update(kv)
        self._log._emit("INFO", self._stage, f"progress {local_pct:.0f}%", **detail)


class Logger:
    """Emits prefixed records for one ``source``.

    ``rank_fn`` returns this process's MPI rank; ``None`` means single-process (rank 0). It is
    called per emission and must itself be collective-free -- see the module docstring.
    """

    def __init__(
        self,
        source: str,
        *,
        rank_fn: Optional[Callable[[], int]] = None,
        stream: Any = None,
        clock: Callable[[], float] = default_clock,
    ) -> None:
        self._source = source
        self._rank_fn = rank_fn
        self._stream = stream
        self._clock = clock

    # -- emission -------------------------------------------------------------------------
    def _rank(self) -> int:
        if self._rank_fn is None:
            return 0
        try:
            return int(self._rank_fn())
        except Exception:
            # A comm that cannot report its rank must not silence the log; assume root and say so.
            if announce_once("heartlog:rank_fn_failed"):
                self._emit_raw(
                    format_line(
                        "WARN", "heartlog", "emit",
                        "rank resolver failed; assuming rank 0 -- records may be duplicated per rank",
                        timestamp=self._clock(),
                    )
                )
            return 0

    def _emit_raw(self, line: str) -> None:
        stream = self._stream if self._stream is not None else sys.stdout
        try:
            print(line, file=stream, flush=True)
        except Exception as exc:
            st = _state()
            if not st.emit_failed:
                st.emit_failed = True
                try:
                    sys.stderr.write(
                        "WARNING [heartlog]: log emission failed (%s: %s); "
                        "continuing WITHOUT log output for this stream -- the run is unaffected "
                        "but its progress is no longer observable.\n" % (type(exc).__name__, exc)
                    )
                    sys.stderr.flush()
                except Exception:
                    pass

    def _emit(self, level: str, stage: str, message: str, /, **kv: Any) -> None:
        if level_rank(level) < level_rank(min_level()):
            return
        rank = self._rank()
        if rank != 0 and not all_ranks_enabled():
            return
        if all_ranks_enabled():
            kv.setdefault("rank", rank)
        self._emit_raw(
            format_line(level, self._source, stage, message, timestamp=self._clock(), **kv)
        )

    # -- public API -----------------------------------------------------------------------
    # `stage`/`message` are positional-only so a caller's own `stage=`/`message=` DETAIL becomes
    # a key=value on the record instead of a TypeError.
    def info(self, stage: str, message: str, /, **kv: Any) -> None:
        self._emit("INFO", stage, message, **kv)

    def warn(self, stage: str, message: str, /, **kv: Any) -> None:
        self._emit("WARN", stage, message, **kv)

    def error(self, stage: str, message: str, /, **kv: Any) -> None:
        self._emit("ERROR", stage, message, **kv)

    def log(self, level: str, stage: str, message: str, /, **kv: Any) -> None:
        """Emit at a level chosen at runtime. Used by the compatibility shims that lift a
        severity word out of a legacy message body."""
        self._emit(str(level).upper(), stage, message, **kv)

    def passthrough(self, line: str) -> None:
        """Write a verbatim, UNPREFIXED line (captured subprocess stdout).

        Rank-gated like any record, but never formatted: the absence of a prefix is precisely
        what marks a line as passthrough on the Raw log.
        """
        if self._rank() != 0 and not all_ranks_enabled():
            return
        self._emit_raw(line.rstrip("\n"))

    def tracker(self, stage: str, /) -> StageCtx:
        """A standalone progress tracker for a loop that cannot be wrapped in :meth:`stage`.

        The coupled time loop is a long ``while`` with several ``break`` paths; wrapping it in a
        context manager would mean reindenting hundreds of lines of solver code for a logging
        change. This yields the same dual-throttled, monotone :meth:`StageCtx.progress` without
        the BEGIN/END bracket, which such a caller emits itself around the loop.
        """
        return StageCtx(self, stage)

    @contextmanager
    def stage(self, name: str, /, **params: Any) -> Iterator[StageCtx]:
        """Bracket one stage with a BEGIN record and exactly one terminal record.

        Exceptions are annotated and **re-raised unchanged**, so the caller keeps ownership of
        the failure -- this never converts a hard failure into a logged warning.
        """
        if not is_known(self._source, name) and announce_once("heartlog:stage:%s/%s" % (self._source, name)):
            self._emit(
                "WARN", "emit",
                "stage name is not in the declared vocabulary for this source; emitting anyway",
                source=self._source, unknown_stage=name,
            )
        ctx = StageCtx(self, name)
        started = self._clock()
        self._emit("INFO", name, "BEGIN", **params)
        try:
            yield ctx
        except BaseException as exc:  # noqa: BLE001 - annotated, then re-raised below.
            elapsed = "%.1fs" % (self._clock() - started)
            code = getattr(exc, "code", None) or exc.__class__.__name__
            detail = str(exc).strip().splitlines()
            self._emit(
                "ERROR", name, "FAIL",
                elapsed=elapsed,
                error_code=code,
                error=(detail[0][:200] if detail else None),
            )
            raise
        else:
            elapsed = "%.1fs" % (self._clock() - started)
            self._emit("INFO", name, "END", elapsed=elapsed, status="ok", **ctx._outputs)


def get_logger(source: str, *, rank_fn: Optional[Callable[[], int]] = None) -> Logger:
    """Convenience constructor. Loggers are cheap and hold no per-instance state worth sharing."""
    return Logger(source, rank_fn=rank_fn)
