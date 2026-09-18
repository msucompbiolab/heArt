import numpy as np

__all__ = [
    "evaluate_full_waveform_periodicity",
    "evaluate_pvloop_periodicity",
    "evaluate_full_waveform_periodicity_from_buffers",
    "evaluate_pvloop_periodicity_from_buffers",
    "write_cycle_summary_metrics",
]


def evaluate_full_waveform_periodicity(csv_path, eps=1e-2, signals=("LV", "RV")):
    """Compare last two cycles of P(t)/V(t) using phase-normalized L2/Linf errors."""
    import csv as _csv

    def _read_columns(path):
        with open(path, "r") as f:
            rdr = _csv.DictReader(f)
            rows = list(rdr)
        if not rows:
            raise ValueError("output_PV.csv has no data rows")
        t = np.array([float(r.get("t", 0.0)) for r in rows], dtype=float)
        data = {
            k: np.array([float(r[k]) for r in rows], dtype=float)
            for k in rows[0].keys()
            if k not in ("tstep", "t")
        }
        return t, data

    def _split_cycles(tvec):
        # output_PV.csv resets `t` to 0 at each cycle; detect boundaries by a decreasing timestep.
        starts = [0]
        for i in range(1, len(tvec)):
            if tvec[i] < tvec[i - 1]:
                starts.append(i)
        if len(starts) < 2:
            raise ValueError("Need at least two cycles to evaluate periodicity")
        i0 = starts[-2]
        i1 = starts[-1]
        i2 = len(tvec)
        if i2 - i1 < 2 or i1 - i0 < 2:
            raise ValueError("Insufficient samples per cycle to compare")
        return (i0, i1), (i1, i2)

    def _interp_norms(t1, x1, t2, x2):
        # Resample both cycles onto a normalized [0,1] phase grid to tolerate variable dt within a beat.
        ngrid = max(len(t1), len(t2))
        grid = np.linspace(0.0, 1.0, ngrid)
        t1n = (t1 - t1[0]) / max(t1[-1] - t1[0], 1e-12)
        t2n = (t2 - t2[0]) / max(t2[-1] - t2[0], 1e-12)
        x1i = np.interp(grid, t1n, x1)
        x2i = np.interp(grid, t2n, x2)
        diff = x2i - x1i
        denom = max(np.max(np.abs(x2i)), 1e-12)
        rel_l2 = np.sqrt(np.mean(diff**2)) / denom
        rel_linf = np.max(np.abs(diff)) / denom
        return rel_l2, rel_linf

    t, data = _read_columns(csv_path)
    (a0, a1), (b0, b1) = _split_cycles(t)

    result = {"ok": True, "signals": {}}
    for tag in signals:
        pkey = f"P_{tag}"
        vkey = f"V_{tag}"
        if pkey not in data or vkey not in data:
            continue
        t1, t2 = t[a0:a1], t[b0:b1]
        p1, p2 = data[pkey][a0:a1], data[pkey][b0:b1]
        v1, v2 = data[vkey][a0:a1], data[vkey][b0:b1]

        p_l2, p_linf = _interp_norms(t1, p1, t2, p2)
        v_l2, v_linf = _interp_norms(t1, v1, t2, v2)

        ok = p_l2 < eps and p_linf < eps and v_l2 < eps and v_linf < eps
        result["ok"] = result["ok"] and ok
        result["signals"][tag] = {
            "P": {"rel_l2": p_l2, "rel_linf": p_linf},
            "V": {"rel_l2": v_l2, "rel_linf": v_linf},
            "ok": ok,
        }

    return result


def evaluate_pvloop_periodicity(csv_path, eps=5e-3, signals=("LV",)):
    """Compare scalar PV features (EDV/ESV/SV/EF/EDP/ESP) across the last two cycles."""
    import csv as _csv

    with open(csv_path, "r") as f:
        rdr = _csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        raise ValueError("output_PV.csv has no data rows")

    t = np.array([float(r.get("t", 0.0)) for r in rows], dtype=float)
    data = {
        k: np.array([float(r[k]) for r in rows], dtype=float)
        for k in rows[0].keys()
        if k not in ("tstep", "t")
    }

    starts = [0]
    for i in range(1, len(t)):
        if t[i] < t[i - 1]:
            starts.append(i)
    if len(starts) < 2:
        raise ValueError("Need at least two cycles to evaluate PV-loop periodicity")
    i0, i1 = starts[-2], starts[-1]
    i2 = len(t)
    if i1 - i0 < 2 or i2 - i1 < 2:
        raise ValueError("Insufficient samples per cycle to compare")

    res = {"ok": True, "signals": {}}
    for tag in signals:
        pkey = f"P_{tag}"
        vkey = f"V_{tag}"
        if pkey not in data or vkey not in data:
            continue
        P1, V1 = data[pkey][i0:i1], data[vkey][i0:i1]
        P2, V2 = data[pkey][i1:i2], data[vkey][i1:i2]

        idx_edv1 = int(np.argmax(V1))
        idx_edv2 = int(np.argmax(V2))
        EDV1, EDV2 = float(V1[idx_edv1]), float(V2[idx_edv2])
        ESV1, ESV2 = float(np.min(V1)), float(np.min(V2))
        SV1, SV2 = EDV1 - ESV1, EDV2 - ESV2
        EF1 = SV1 / max(EDV1, 1e-12)
        EF2 = SV2 / max(EDV2, 1e-12)
        EDP1, EDP2 = float(P1[idx_edv1]), float(P2[idx_edv2])
        ESP1, ESP2 = float(np.max(P1)), float(np.max(P2))

        def rel(a, b):
            return abs(a - b) / max(abs(b), 1e-12)

        diffs = {
            "EDV": rel(EDV2, EDV1),
            "ESV": rel(ESV2, ESV1),
            "SV": rel(SV2, SV1),
            "EF": rel(EF2, EF1),
            "EDP": rel(EDP2, EDP1),
            "ESP": rel(ESP2, ESP1),
        }
        ok = all(v < eps for v in diffs.values())
        res["ok"] = res["ok"] and ok
        res["signals"][tag] = {"diffs": diffs, "ok": ok}

    return res


def _interp_norms_arrays(t1, x1, t2, x2):
    ngrid = max(len(t1), len(t2))
    grid = np.linspace(0.0, 1.0, max(ngrid, 2))
    t1n = (np.array(t1) - t1[0]) / max(t1[-1] - t1[0], 1e-12)
    t2n = (np.array(t2) - t2[0]) / max(t2[-1] - t2[0], 1e-12)
    x1i = np.interp(grid, t1n, x1)
    x2i = np.interp(grid, t2n, x2)
    diff = x2i - x1i
    denom = max(np.max(np.abs(x2i)), 1e-12)
    rel_l2 = np.sqrt(np.mean(diff**2)) / denom
    rel_linf = np.max(np.abs(diff)) / denom
    return rel_l2, rel_linf


def evaluate_full_waveform_periodicity_from_buffers(
    prev_buf, curr_buf, eps=1e-2, signals=("LV", "RV")
):
    # Same metric as evaluate_full_waveform_periodicity(), but avoids CSV parsing in long runs.
    """Compute waveform periodicity from in-memory cycle buffers."""
    result = {"ok": True, "signals": {}}
    t1, t2 = np.array(prev_buf["t"]), np.array(curr_buf["t"])
    if len(t1) < 2 or len(t2) < 2:
        raise ValueError("Insufficient samples per cycle to compare")
    for tag in signals:
        pkey = f"P_{tag}"
        vkey = f"V_{tag}"
        if pkey not in prev_buf or vkey not in prev_buf:
            continue
        p1, p2 = np.array(prev_buf[pkey]), np.array(curr_buf[pkey])
        v1, v2 = np.array(prev_buf[vkey]), np.array(curr_buf[vkey])
        p_l2, p_linf = _interp_norms_arrays(t1, p1, t2, p2)
        v_l2, v_linf = _interp_norms_arrays(t1, v1, t2, v2)
        ok = p_l2 < eps and p_linf < eps and v_l2 < eps and v_linf < eps
        result["ok"] = result["ok"] and ok
        result["signals"][tag] = {
            "P": {"rel_l2": p_l2, "rel_linf": p_linf},
            "V": {"rel_l2": v_l2, "rel_linf": v_linf},
            "ok": ok,
        }
    return result


def evaluate_pvloop_periodicity_from_buffers(
    prev_buf, curr_buf, eps=5e-3, signals=("LV",)
):
    """Compute PV-loop periodicity from in-memory cycle buffers."""
    res = {"ok": True, "signals": {}}
    for tag in signals:
        pkey = f"P_{tag}"
        vkey = f"V_{tag}"
        if pkey not in prev_buf or vkey not in prev_buf:
            continue
        P1, V1 = np.array(prev_buf[pkey]), np.array(prev_buf[vkey])
        P2, V2 = np.array(curr_buf[pkey]), np.array(curr_buf[vkey])
        if len(P1) < 2 or len(P2) < 2:
            raise ValueError("Insufficient samples per cycle to compare")
        idx_edv1 = int(np.argmax(V1))
        idx_edv2 = int(np.argmax(V2))
        EDV1, EDV2 = float(V1[idx_edv1]), float(V2[idx_edv2])
        ESV1, ESV2 = float(np.min(V1)), float(np.min(V2))
        SV1, SV2 = EDV1 - ESV1, EDV2 - ESV2
        EF1 = SV1 / max(EDV1, 1e-12)
        EF2 = SV2 / max(EDV2, 1e-12)
        EDP1, EDP2 = float(P1[idx_edv1]), float(P2[idx_edv2])
        ESP1, ESP2 = float(np.max(P1)), float(np.max(P2))
        rel = lambda a, b: abs(a - b) / max(abs(b), 1e-12)
        diffs = {
            "EDV": rel(EDV2, EDV1),
            "ESV": rel(ESV2, ESV1),
            "SV": rel(SV2, SV1),
            "EF": rel(EF2, EF1),
            "EDP": rel(EDP2, EDP1),
            "ESP": rel(ESP2, ESP1),
        }
        ok = all(v < eps for v in diffs.values())
        res["ok"] = res["ok"] and ok
        res["signals"][tag] = {"diffs": diffs, "ok": ok}
    return res


def write_cycle_summary_metrics(csv_path, out_csv_path=None):
    """Write a one-row CSV summary from the last complete cycle (LV/RV PV + coarse LA wave metrics)."""
    import csv as _csv

    def _read(path):
        with open(path, "r") as f:
            rdr = _csv.DictReader(f)
            rows = list(rdr)
        if not rows:
            raise ValueError("output_PV.csv has no data rows")
        t = np.array([float(r.get("t", 0.0)) for r in rows], dtype=float)
        data = {
            k: np.array([float(r[k]) for r in rows], dtype=float)
            for k in rows[0].keys()
            if k not in ("tstep", "t")
        }
        return t, data

    def _last_cycle_idx(t):
        starts = [0]
        for i in range(1, len(t)):
            if t[i] < t[i - 1]:
                starts.append(i)
        if len(starts) < 1:
            raise ValueError("Could not determine cycle boundaries")
        i1 = starts[-1]
        i2 = len(t)
        if i2 - i1 < 2 and len(starts) >= 2:
            i1 = starts[-2]
        return i1, len(t)

    def _local_peaks(x):
        if len(x) < 3:
            return []
        return [i for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] > x[i + 1]]

    def _local_mins(x):
        if len(x) < 3:
            return []
        return [i for i in range(1, len(x) - 1) if x[i] < x[i - 1] and x[i] < x[i + 1]]

    t, data = _read(csv_path)
    i1, i2 = _last_cycle_idx(t)
    tcyc = t[i1:i2]

    # Pressures in output_PV.csv are in Pa; convert to mmHg for summary metrics
    pa_to_mmhg = 0.0075

    LV_EDV = LV_ESV = LV_SV = LV_EF = LV_P_sys_peak = LV_P_EDP = ""
    if "V_LV" in data and "P_LV" in data:
        V = data["V_LV"][i1:i2]
        P = data["P_LV"][i1:i2] * pa_to_mmhg
        if len(V) >= 2:
            idx_edv = int(np.argmax(V))
            LV_EDV = float(V[idx_edv])
            LV_ESV = float(np.min(V))
            LV_SV = float(LV_EDV - LV_ESV)
            LV_EF = float(LV_SV / LV_EDV) if LV_EDV else 0.0
            LV_P_sys_peak = float(np.max(P))
            LV_P_EDP = float(P[idx_edv])

    RV_EDV = RV_ESV = RV_SV = RV_EF = RV_P_sys_peak = RV_P_EDP = ""
    if "V_RV" in data and "P_RV" in data:
        V = data["V_RV"][i1:i2]
        P = data["P_RV"][i1:i2] * pa_to_mmhg
        if len(V) >= 2:
            idx_edv = int(np.argmax(V))
            RV_EDV = float(V[idx_edv])
            RV_ESV = float(np.min(V))
            RV_SV = float(RV_EDV - RV_ESV)
            RV_EF = float(RV_SV / RV_EDV) if RV_EDV else 0.0
            RV_P_sys_peak = float(np.max(P))
            RV_P_EDP = float(P[idx_edv])

    LA_Vmax = LA_Vmin = LA_VpreA = LA_P_mean = LA_P_v_peak = LA_P_a_amp = LA_P_a_dur = LA_t_v_peak = ""
    if "V_LA" in data and "P_LA" in data:
        V = data["V_LA"][i1:i2]
        P = data["P_LA"][i1:i2] * pa_to_mmhg
        if len(V) >= 2:
            LA_Vmax = float(np.max(V))
            LA_Vmin = float(np.min(V))
            LA_P_mean = float(np.mean(P))
            peaks = _local_peaks(P)
            mins = _local_mins(P)
            if peaks:
                # Heuristic separation: treat early peak as a-wave and late peak as v-wave.
                early_window = int(0.6 * len(P))
                a_candidates = [i for i in peaks if i < early_window]
                a_idx = a_candidates[0] if a_candidates else peaks[0]
                pre_mins = [m for m in mins if m < a_idx]
                pre_min = pre_mins[-1] if pre_mins else 0
                LA_P_a_amp = float(max(P[a_idx] - P[pre_min], 0.0))
                post_mins = [m for m in mins if m > a_idx]
                post_min = post_mins[0] if post_mins else a_idx
                LA_P_a_dur = float(tcyc[post_min] - tcyc[pre_min])
                LA_VpreA = float(V[pre_min])
                late_start = int(0.6 * len(P))
                v_candidates = [i for i in peaks if i >= late_start]
                if v_candidates:
                    v_idx = max(v_candidates, key=lambda i: P[i])
                else:
                    v_idx = int(np.argmax(P))
                LA_P_v_peak = float(P[v_idx])
                LA_t_v_peak = float(tcyc[v_idx] - tcyc[0])

    if out_csv_path is None:
        import os as _os

        out_csv_path = _os.path.join(_os.path.dirname(csv_path), "baseline_summary.csv")

    headers = [
        "LV_EDV",
        "LV_ESV",
        "LV_SV",
        "LV_EF",
        "LV_P_sys_peak",
        "LV_P_EDP",
        "RV_EDV",
        "RV_ESV",
        "RV_SV",
        "RV_EF",
        "RV_P_sys_peak",
        "RV_P_EDP",
        "LA_Vmax",
        "LA_Vmin",
        "LA_VpreA",
        "LA_P_mean",
        "LA_P_v_peak",
        "LA_P_a_amp",
        "LA_P_a_dur",
        "LA_t_v_peak",
    ]
    row = [
        LV_EDV,
        LV_ESV,
        LV_SV,
        LV_EF,
        LV_P_sys_peak,
        LV_P_EDP,
        RV_EDV,
        RV_ESV,
        RV_SV,
        RV_EF,
        RV_P_sys_peak,
        RV_P_EDP,
        LA_Vmax,
        LA_Vmin,
        LA_VpreA,
        LA_P_mean,
        LA_P_v_peak,
        LA_P_a_amp,
        LA_P_a_dur,
        LA_t_v_peak,
    ]

    with open(out_csv_path, "w", newline="") as f:
        wr = _csv.writer(f)
        wr.writerow(headers)
        wr.writerow(row)

    return out_csv_path
