"""OPEN-LOOP volume driving of the four-chamber FE model from a 0D PV trace.

dolfin-free. Under ``SimDet["fch_coupling_mode"] == "openloop_volume"`` the driver does NOT
advance the 0D circulation: the four chamber volumes are READ from an established
``output_PV.csv`` (the re-anchored 0D forward), imposed on the FE model through the
monolithic swept-cavity constraints, and the FE cavity pressures are reported but never fed
back. It is the one-way half of the coupling the HFpEF LV+aorta manuscript describes
("the circulation ... returns a target cavity volume, which is imposed as a constraint in the
finite-element solve; ventricular pressure is obtained as the associated Lagrange
multiplier"), with the circulation replaced by its own recorded trace.

WHY it exists: it yields the full deformation history of a cycle -- every Figure 6 frame from
ONE run -- with no coupling failure mode, and it measures, per chamber, how far the FE
operating point sits from the 0D loop's (the FE pressure at the imposed volume against the
trace's pressure). Every artifact it produces is PRELIMINARY (open-loop) and is labelled so.

The trace is the LAST COMPLETE CYCLE of the CSV (cycle-local ``t`` resets each beat: the
convention the periodicity code relies on), replayed PERIODICALLY: ``at(t)`` interpolates
linearly on ``t mod BCL`` and wraps the segment between the last sample and the first sample
one beat later, so a run of N cycles replays the same beat N times.
"""
from __future__ import annotations

import bisect
import csv
import math
from typing import Dict, List, Sequence

CHAMBERS = ("LV", "RV", "LA", "RA")
VOLUME_KEYS = tuple("V_%s" % ch for ch in CHAMBERS)
PRESSURE_KEYS = tuple("P_%s" % ch for ch in CHAMBERS)


def read_trace(path: str) -> List[Dict[str, float]]:
    """Parse an ``output_PV.csv`` (header row; every column numeric) into float row dicts."""
    with open(path, newline="") as fh:
        rows = []
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                if k is None or v in ("", None):
                    continue
                try:
                    row[k.strip()] = float(v)
                except ValueError:
                    continue
            rows.append(row)
    if not rows:
        raise ValueError("empty PV trace: %s" % path)
    missing = [k for k in ("t",) + VOLUME_KEYS if k not in rows[0]]
    if missing:
        raise ValueError("PV trace %s lacks required column(s) %s" % (path, missing))
    return rows


def last_complete_cycle(rows: Sequence[Dict[str, float]], bcl_ms: float) -> List[Dict[str, float]]:
    """Rows of the last COMPLETE beat (a boundary is a decrease in cycle-local ``t``)."""
    starts = [0] + [i for i in range(1, len(rows)) if rows[i]["t"] < rows[i - 1]["t"]]
    bounds = list(zip(starts, starts[1:] + [len(rows)]))
    dt = rows[1]["t"] - rows[0]["t"] if len(rows) > 1 else 0.0
    need = float(bcl_ms) - 2.0 * max(dt, 0.0)
    for lo, hi in reversed(bounds):
        seg = rows[lo:hi]
        if len(seg) > 1 and (seg[-1]["t"] - seg[0]["t"]) >= need:
            return list(seg)
    raise ValueError(
        "PV trace contains no complete %.0f ms cycle (%d segment(s), longest span %.1f ms)"
        % (bcl_ms, len(bounds),
           max((rows[hi - 1]["t"] - rows[lo]["t"]) for lo, hi in bounds)))


class VolumeTrace:
    """Periodic interpolant over one complete beat of a 0D PV trace.

    ``at(t_ms)`` returns every numeric column interpolated at ``t mod BCL``; ``volumes(t_ms)``
    returns the four chamber volumes in driver order (LV, RV, LA, RA).
    """

    def __init__(self, path: str, bcl_ms: float):
        self.path = str(path)
        self.bcl_ms = float(bcl_ms)
        if not (self.bcl_ms > 0.0):
            raise ValueError("bcl_ms must be positive, got %r" % bcl_ms)
        rows = read_trace(self.path)
        cycle = last_complete_cycle(rows, self.bcl_ms)
        t0 = cycle[0]["t"]
        self._t = [float(r["t"] - t0) for r in cycle]
        if any(b <= a for a, b in zip(self._t, self._t[1:])):
            raise ValueError("PV trace cycle is not strictly increasing in t")
        keys = set(cycle[0].keys())
        for r in cycle[1:]:
            keys &= set(r.keys())
        self.columns = tuple(sorted(k for k in keys if k != "t"))
        self._cols = {k: [float(r[k]) for r in cycle] for k in self.columns}
        self.n_rows = len(cycle)
        self.source_rows = len(rows)

    # -- lookups ------------------------------------------------------------------
    def _phase(self, t_ms: float) -> float:
        tau = math.fmod(float(t_ms), self.bcl_ms)
        return tau + self.bcl_ms if tau < 0.0 else tau

    def at(self, t_ms: float) -> Dict[str, float]:
        tau = self._phase(t_ms)
        t = self._t
        n = len(t)
        # wrap: between the last sample and the first sample one beat later
        if tau >= t[-1]:
            span = (t[0] + self.bcl_ms) - t[-1]
            w = 0.0 if span <= 0.0 else (tau - t[-1]) / span
            return {k: (1.0 - w) * self._cols[k][-1] + w * self._cols[k][0] for k in self.columns}
        if tau <= t[0]:
            return {k: self._cols[k][0] for k in self.columns}
        i = bisect.bisect_right(t, tau)          # t[i-1] <= tau < t[i]
        a, b = t[i - 1], t[i]
        w = (tau - a) / (b - a) if b > a else 0.0
        return {k: (1.0 - w) * self._cols[k][i - 1] + w * self._cols[k][i] for k in self.columns}

    def volumes(self, t_ms: float):
        row = self.at(t_ms)
        return tuple(row[k] for k in VOLUME_KEYS)

    def pressures_pa(self, t_ms: float):
        row = self.at(t_ms)
        return tuple(row.get(k, float("nan")) for k in PRESSURE_KEYS)

    def cycle_start_volumes(self) -> Dict[str, float]:
        """The chamber volumes at the beat's first sample (cycle-local t = 0 = end-diastole)."""
        return {ch: self._cols["V_%s" % ch][0] for ch in CHAMBERS}

    def describe(self) -> Dict[str, object]:
        v0 = self.cycle_start_volumes()
        return {
            "path": self.path, "bcl_ms": self.bcl_ms, "cycle_rows": self.n_rows,
            "source_rows": self.source_rows,
            "cycle_start_volumes_ml": v0,
            "volume_range_ml": {ch: [min(self._cols["V_%s" % ch]), max(self._cols["V_%s" % ch])]
                                for ch in CHAMBERS},
        }
