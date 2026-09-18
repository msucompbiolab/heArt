"""Volume-prescribed preload ramp for the LVW volume-control coupling mode.

Under ``SimDet["lvw_swept_volctrl"]`` the LVW mechanics model carries a monolithic
swept-cavity volume constraint whose Lagrange multiplier IS the cavity pressure; the
prescribed-pressure Constant is not in the residual, so ``run_waorta``'s pressure ramp
to EDP cannot load the reference. This module marches the PRESCRIBED VOLUME instead and
reads the pressure off the multiplier until it reaches EDP, mirroring the adaptive
inner loop of ``MEmodel.unloading`` (halve the increment on a solve failure or a
pressure overshoot; the target is bracketed by two SOLVED states and never
interpolated).

The ramp is dolfin-free and driven through callbacks so it is unit-testable with a
synthetic monotone P(V); the driver supplies the FE solve, snapshot and restore.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional


class VolumeRampError(RuntimeError):
    """The volume ramp could not reach its target (increment floor hit or solve budget out)."""


def ramp_volume_to_pressure(
    *,
    v_start: float,
    p_start: float,
    p_target: float,
    solve_at: Callable[[float], float],
    snapshot: Callable[[], Any],
    restore: Callable[[Any], None],
    volinc: float,
    tol_p: float,
    volinc_floor: float = 1.0e-4,
    max_solves: int = 500,
    stop_volume: Optional[float] = None,
    p_cap: Optional[float] = None,
    on_step: Optional[Callable[[float, float], None]] = None,
    warn: Optional[Callable[[str], None]] = None,
) -> dict:
    """March the prescribed cavity volume from ``v_start`` (an already SOLVED state at
    pressure ``p_start``) until the multiplier pressure reaches ``p_target`` within
    ``tol_p`` (same units as the pressures), or -- when ``stop_volume`` is given -- until
    the prescribed volume reaches it (the volume-anchored preload the pressure path
    offers through ``preload_target_volume_ml``).

    ``solve_at(v)`` prescribes ``v``, solves, and returns the multiplier pressure; it
    RAISES on a solver failure. ``snapshot()``/``restore(obj)`` capture and roll back the
    FE state. Every accepted state is reported through ``on_step(v, p)``.

    Policy (mirrors ``MEmodel.unloading``'s inner loop):
      * solve failure  -> restore the last good state, halve ``volinc``, retry;
      * pressure overshoot past ``p_target + tol_p`` -> restore, halve, retry
        (a bisection on the last interval, so the final state is SOLVED at the target);
      * ``volinc`` below ``volinc_floor`` or more than ``max_solves`` solves -> raise
        :class:`VolumeRampError` (no silent partial preload).
    In volume-anchored mode ``p_cap`` is the pressure safety cap the pressure path also
    applies (stop with a warning if the target volume is unreachable below it).
    """
    if volinc <= 0.0:
        raise ValueError("volinc must be positive (the direction is derived from the target)")
    if tol_p <= 0.0:
        raise ValueError("tol_p must be positive")
    if stop_volume is None and not math.isfinite(p_target):
        raise ValueError("p_target must be finite")

    _warn = warn or (lambda _msg: None)
    if stop_volume is not None:
        direction = 1.0 if stop_volume >= v_start else -1.0
    else:
        direction = 1.0 if p_target >= p_start else -1.0
        if abs(p_target - p_start) <= tol_p:
            _warn("volume ramp: the zero-volume state already sits at the target pressure "
                  "(p=%.6g, target %.6g); no ramp performed" % (p_start, p_target))
            return {"v": v_start, "p": p_start, "n_solves": 0, "n_halvings": 0,
                    "n_accepted": 0, "volinc_final": volinc, "reason": "already_at_target"}
        if direction < 0.0:
            _warn("volume ramp: zero-volume pressure %.6g exceeds the target %.6g; "
                  "ramping the volume DOWN" % (p_start, p_target))

    v_good, p_good = float(v_start), float(p_start)
    state_good = snapshot()
    n_solves = n_halvings = n_accepted = 0
    reason = ""
    while True:
        if n_solves >= max_solves:
            raise VolumeRampError(
                "volume ramp exhausted its solve budget (%d) at V=%.6g, P=%.6g "
                "(target P=%.6g)" % (max_solves, v_good, p_good, p_target))
        step = volinc
        if stop_volume is not None:
            step = min(step, abs(stop_volume - v_good))
        v_try = v_good + direction * step
        try:
            p_try = float(solve_at(v_try))
            n_solves += 1
            if not math.isfinite(p_try):
                raise VolumeRampError("non-finite multiplier pressure at V=%.6g" % v_try)
        except Exception as exc:  # the FE solve is the only thing that raises here
            n_solves += 1
            restore(state_good)
            volinc *= 0.5
            n_halvings += 1
            _warn("volume ramp: solve FAILED at V=%.6g (%s: %s); rolled back to V=%.6g and "
                  "halved the increment to %.6g mL" % (v_try, type(exc).__name__, exc,
                                                        v_good, volinc))
            if volinc < volinc_floor:
                raise VolumeRampError(
                    "volume ramp: increment fell below the floor %g mL at V=%.6g, P=%.6g "
                    "(target P=%.6g) -- the ramp cannot pass this state"
                    % (volinc_floor, v_good, p_good, p_target)) from exc
            continue

        if stop_volume is None and direction * (p_try - p_target) > tol_p:
            # Overshoot: bisect the last interval from the last good SOLVED state.
            restore(state_good)
            volinc *= 0.5
            n_halvings += 1
            if volinc < volinc_floor:
                raise VolumeRampError(
                    "volume ramp: pressure overshoot could not be bisected below the "
                    "increment floor %g mL (V=%.6g P=%.6g, overshoot P=%.6g, target %.6g "
                    "tol %.3g)" % (volinc_floor, v_good, p_good, p_try, p_target, tol_p))
            continue

        v_good, p_good = v_try, p_try
        state_good = snapshot()
        n_accepted += 1
        if on_step is not None:
            on_step(v_good, p_good)

        if stop_volume is not None:
            if direction * (v_good - stop_volume) >= -1.0e-12:
                reason = "volume_reached"
                break
            if p_cap is not None and p_good >= p_cap:
                _warn("volume ramp: pressure cap %.6g reached at V=%.6g before the target "
                      "volume %.6g; stopping" % (p_cap, v_good, stop_volume))
                reason = "pressure_cap"
                break
        elif abs(p_good - p_target) <= tol_p:
            reason = "pressure_reached"
            break

    return {"v": v_good, "p": p_good, "n_solves": n_solves, "n_halvings": n_halvings,
            "n_accepted": n_accepted, "volinc_final": volinc, "reason": reason}


def ramp_volumes_to_pressures(
    *,
    v_start: dict,
    p_targets: dict,
    solve_at: Callable[[dict], dict],
    snapshot: Callable[[], Any],
    restore: Callable[[Any], None],
    volinc: float,
    tol_p: float,
    volinc_floor: float = 1.0e-4,
    max_solves: int = 600,
    on_step: Optional[Callable[[dict, dict], None]] = None,
    warn: Optional[Callable[[str], None]] = None,
) -> dict:
    """FOUR-cavity generalisation of :func:`ramp_volume_to_pressure` (the fch_coupled rung-U
    loading of an UNLOADED four-chamber reference under ``volctrl4``).

    ``v_start`` = {ch: prescribed volume of the already-SOLVED zero state}; ``p_targets`` =
    {ch: EDP} (same pressure units the multipliers report). ``solve_at(vols)`` prescribes every
    chamber's volume, solves ONCE (all four constraints at once) and returns {ch: multiplier};
    it RAISES on a solver failure. Every chamber marches by the SHARED increment ``volinc``
    until its multiplier is inside ``tol_p`` of its target, then it is FROZEN at that volume
    while the others continue; a chamber that overshoots is bisected on its own last interval
    (its increment halves, the others keep marching -- the four cavities are coupled through
    the septa and the AV plane, so re-solving with the frozen ones held keeps the state solved).
    A solve failure restores the last good state and halves the shared increment. Floor /
    budget -> :class:`VolumeRampError` (never a silent partial preload).
    """
    chs = list(p_targets)
    if volinc <= 0.0 or tol_p <= 0.0:
        raise ValueError("volinc and tol_p must be positive")
    _warn = warn or (lambda _msg: None)
    v_good = {c: float(v_start[c]) for c in chs}
    inc = {c: float(volinc) for c in chs}
    state_good = snapshot()
    p_good = solve_at(dict(v_good))          # the zero state, solved (u = 0 -> multipliers ~0)
    n_solves, n_halvings, n_accepted = 1, 0, 0
    def _active():
        # A chamber marches whenever its multiplier is outside the tolerance -- ALSO after it
        # was at target, because the septa/AV plane couple the cavities and a neighbour's
        # inflation moves it again. Termination = every chamber inside tol at once.
        return [c for c in chs if abs(p_good[c] - p_targets[c]) > tol_p]

    if not _active():
        _warn("four-cavity ramp: the zero state already sits at every target pressure")
        return {"v": v_good, "p": p_good, "n_solves": n_solves, "n_halvings": 0,
                "n_accepted": 0, "reason": "already_at_target"}
    while True:
        act = _active()
        if not act:
            break
        if n_solves >= max_solves:
            raise VolumeRampError("four-cavity ramp exhausted its solve budget (%d) at V=%s P=%s"
                                  % (max_solves, v_good, p_good))
        if min(inc[c] for c in act) < volinc_floor:
            raise VolumeRampError("four-cavity ramp: an increment fell below the floor %g at "
                                  "V=%s P=%s (targets %s)" % (volinc_floor, v_good, p_good, p_targets))
        sgn = {c: (1.0 if p_targets[c] >= p_good[c] else -1.0) for c in chs}
        v_try = {c: (v_good[c] + sgn[c] * inc[c] if c in act else v_good[c]) for c in chs}
        try:
            p_try = solve_at(dict(v_try))
            n_solves += 1
            if not all(math.isfinite(p_try[c]) for c in chs):
                raise VolumeRampError("non-finite multiplier at V=%s" % v_try)
        except Exception as exc:
            n_solves += 1
            restore(state_good)
            for c in act:
                inc[c] *= 0.5
            n_halvings += 1
            _warn("four-cavity ramp: solve FAILED at V=%s (%s: %s); rolled back, increments halved"
                  % ({c: round(v, 4) for c, v in v_try.items()}, type(exc).__name__, exc))
            continue
        # a marching chamber that overshoots its target bisects ITS interval; the rest keep going
        over = [c for c in act if sgn[c] * (p_try[c] - p_targets[c]) > tol_p]
        if over:
            restore(state_good)
            for c in over:
                inc[c] *= 0.5
            n_halvings += 1
            continue
        v_good, p_good, state_good = v_try, p_try, snapshot()
        n_accepted += 1
        if on_step is not None:
            on_step(dict(v_good), dict(p_good))
    return {"v": v_good, "p": p_good, "n_solves": n_solves, "n_halvings": n_halvings,
            "n_accepted": n_accepted, "reason": "at_target"}


def volctrl_edp_targets_pa(edp_mmhg: float, p_lv_pa: float, p_rv_pa: float, p_la_pa: float,
                           p_ra_pa: float, mmhg_per_pa: float = 0.0075) -> dict:
    """Per-chamber END-DIASTOLIC pressure targets (Pa) for the four-cavity volume ramp.

    The pressure loading path increments all four cavity pressures in LOCKSTEP by
    ``P_*/nLoadSteps`` until the LV reaches ``EDP`` (mmHg), so at the end the LV sits at EDP and
    every other chamber at ``EDP * P_ch / P_LV``. The volume ramp must land on the SAME state.
    MEASURED defect (forward v5 arm 0, 2026-09-10): the ramp targeted ``SimDet["P_LV"]`` itself --
    the loading INCREMENT, 6.15 mmHg on the anchored set -- instead of EDP 8, and the other
    chambers followed at 4.6/3.1/3.1 instead of 6/4/4.
    """
    if edp_mmhg <= 0.0 or p_lv_pa <= 0.0:
        raise ValueError("EDP and P_LV must be positive (EDP=%r mmHg, P_LV=%r Pa)" % (edp_mmhg, p_lv_pa))
    lv = float(edp_mmhg) / mmhg_per_pa
    return {"lv": lv, "rv": lv * float(p_rv_pa) / float(p_lv_pa),
            "la": lv * float(p_la_pa) / float(p_lv_pa), "ra": lv * float(p_ra_pa) / float(p_lv_pa)}
