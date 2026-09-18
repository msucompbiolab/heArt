"""Event-time displacement SNAPSHOTS of a four-chamber run, stored as HDF5 + XDMF.

LEGACY FEniCS ONLY. The manuscript's Figure 6 shows the deformation at NAMED cardiac events
(mitral closure, peak LV pressure, tricuspid opening, mitral opening, atrial systole). A
forward run (closed-loop or open-loop) crosses those cycle-local times every beat; this writer
captures the displacement field the first time each event is crossed in every cycle at or
after ``first_cycle``, so the last complete cycle's frames are always on disk.

Two artifacts per snapshot, in the exact layout ``fch_events/export.py`` established so the
Figure 6 renderer (``python -m fch_events render``) and the service node's Mesh View
(``vtkXdmfReader``) read them unchanged:

* ``snapshot_<key>_c<cycle>.h5``       -- DOLFIN ``mesh``, ``u`` and the mixed state ``w_me``
                                          (collective write; survives MPI).
* ``snapshot_<key>_c<cycle>_viz.xdmf`` -- the displacement as an XDMF entry (DISTINCT basename:
                                          XDMFFile writes its data to <basename>.h5 and would
                                          otherwise clobber the HDF5File above).
* ``snapshots.json``                   -- rank-0 index: per snapshot the event key, cycle,
                                          cycle-local time actually captured, and the file paths.

Deliberately NOT routed through ``src/utils/oops_objects_MRC2.exportfiles`` (it deletes every
``*.pvd/*.vtu/*.hdf5/*.xdmf`` in its directory on construction and opens ``Data.h5``
truncating). The snapshot directory is its own subdirectory of the run output.

The event list is read from a ``fch_events`` event table (``python -m fch_events events``:
``{"events": [{"key", "t_ms", ...}, ...]}``) or from a plain ``{key: t_ms}`` mapping, so no
event time is ever typed here.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import dolfin as df


def load_event_times(spec: Any) -> Dict[str, float]:
    """``{key: cycle-local t_ms}`` from an event-table JSON path, an event-table dict, or a
    plain mapping. Rejects anything without at least one finite time."""
    if isinstance(spec, str):
        with open(spec) as fh:
            spec = json.load(fh)
    if not isinstance(spec, dict):
        raise ValueError("event snapshot spec must be a mapping or an event-table JSON path")
    out: Dict[str, float] = {}
    if "events" in spec and isinstance(spec["events"], list):
        for ev in spec["events"]:
            key, t = ev.get("key"), ev.get("t_ms")
            if key is None or t is None:
                continue
            out[str(key)] = float(t)
    else:
        for key, t in spec.items():
            if key.startswith("_"):
                continue
            try:
                out[str(key)] = float(t)
            except (TypeError, ValueError):
                continue
    if not out:
        raise ValueError("event snapshot spec carries no (key, t_ms) pairs")
    return out


class EventSnapshotWriter:
    """Capture the displacement field when the run's cycle-local clock crosses an event time."""

    def __init__(self, out_dir: str, events: Dict[str, float], comm, *, bcl_ms: float,
                 first_cycle: int = 1, log=None):
        self.out_dir = str(out_dir)
        self.events = {str(k): float(v) for k, v in events.items()}
        self.comm = comm
        self.bcl_ms = float(bcl_ms)
        self.first_cycle = int(first_cycle)
        self.log = log
        self._done: set = set()          # (key, cycle) already written
        self.index: List[Dict[str, Any]] = []
        if df.MPI.rank(comm) == 0:
            os.makedirs(self.out_dir, exist_ok=True)

    def due(self, t_prev: float, t_now: float, cycle: int, cycle_prev: int) -> List[str]:
        """Event keys crossed by advancing the cycle-local clock from ``t_prev`` (previous
        step, cycle ``cycle_prev``) to ``t_now`` (cycle ``cycle``). A wrap (new cycle) closes
        the previous beat and opens the next: an event at 0 ms fires on the first sample of a
        beat."""
        if cycle < self.first_cycle:
            return []
        keys = []
        for key, te in self.events.items():
            if (key, cycle) in self._done:
                continue
            if cycle != cycle_prev:
                # crossed the beat boundary: t_prev (late in the old beat) -> t_now (early)
                hit = (te >= 0.0 and te <= t_now)
            else:
                hit = (t_prev < te <= t_now)
            if hit:
                keys.append(key)
        return keys

    def write(self, ME, key: str, cycle: int, t_now: float, tstep: float) -> Dict[str, str]:
        tag = "%s_c%d" % (key, int(cycle))
        u = ME.get_displacement()
        u.rename("displacement", "displacement")
        h5_path = os.path.join(self.out_dir, "snapshot_%s.h5" % tag)
        with df.HDF5File(self.comm, h5_path, "w") as h5:
            h5.write(ME.mesh_me, "mesh")
            h5.write(u, "u")
            h5.write(ME.w_me, "w_me")
        xdmf_path = os.path.join(self.out_dir, "snapshot_%s_viz.xdmf" % tag)
        xdmf = df.XDMFFile(self.comm, xdmf_path)
        xdmf.parameters["functions_share_mesh"] = True
        xdmf.parameters["rewrite_function_mesh"] = False
        xdmf.parameters["flush_output"] = True
        xdmf.write(u, float(t_now))
        xdmf.close()
        self._done.add((key, int(cycle)))
        rec = {"key": key, "cycle": int(cycle), "t_ms": float(t_now), "tstep_ms": float(tstep),
               "target_t_ms": self.events[key], "h5": h5_path, "xdmf": xdmf_path,
               "xdmf_h5": os.path.join(self.out_dir, "snapshot_%s_viz.h5" % tag),
               "reference": "unloaded_or_no_preload_reference (u is the FULL displacement)"}
        self.index.append(rec)
        if df.MPI.rank(self.comm) == 0:
            with open(os.path.join(self.out_dir, "snapshots.json"), "w") as fh:
                json.dump({"schema": 1, "provenance": "run_light.fch_snapshots",
                           "events": self.events, "bcl_ms": self.bcl_ms,
                           "snapshots": self.index}, fh, indent=2)
        if self.log is not None:
            self.log.info("snapshot", "event snapshot written", event=key, cycle=int(cycle),
                          t_ms=round(float(t_now), 3), target_t_ms=self.events[key],
                          xdmf=os.path.basename(xdmf_path))
        return rec

    def maybe_write(self, ME, *, t_prev: float, t_now: float, cycle: int, cycle_prev: int,
                    tstep: float) -> List[Dict[str, str]]:
        """Collective: every rank must call this with identical arguments."""
        written = []
        for key in self.due(t_prev, t_now, cycle, cycle_prev):
            written.append(self.write(ME, key, cycle, t_now, tstep))
        return written
